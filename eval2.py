import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


# Parameters
DATA_FOLDER = './json_folder' # same as output folder for create_input_files.py
DATA_NAME = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files 

CHECKPOINT = './checkpoints/BEST_checkpt_9_epoch.pt'

WORD_MAP_FILE = './json_folder/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
DEVICE = torch.device('cuda',0)
cudnn.benchmark = True # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


# Load Model
checkpoint = torch.load(CHECKPOINT)
decoder = checkpoint['decoder'].to(DEVICE)
decoder.eval()

encoder = checkpoint['encoder'].to(DEVICE)
encoder.eval()

# Load word map
with open(WORD_MAP_FILE, 'r') as j:
  word_map = json.load(j)

rev_word_map = {v:k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalize transform
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])


def evaluate(beam_size):
  '''
  Evaluation

  -- beam_size : beam size at which to generate captions for evaluation
  return: BLEU-4 score
  '''

  # Dataloader
  loader = torch.utils.data.DataLoader(
      Caption_Dataset(DATA_FOLDER, DATA_NAME, 'TEST', state = 'TEST', transform = transforms. Compose([normalize])),
      batch_size =1, shuffle = True, num_workers= 1, pin_memory=True
  )
  
  # Becasue we are doing beam search, so do NOT use batch size > 1 - IMPORTANT! 

  # Lists to store references (true captions), and hypothesis (predictions) for each image
  # If for n images, we have n hypothesis, and reference a,b,c for each image, we need
  # reference = [[ref1a, ref1b, ref1c], [ref2a, ref2b, ref2c], .. ], hypothesis = [hyp1, hyp2, ..]
  references = list()
  hypothesis = list()

  # for each image
  for i, (image, caps, caplens, allcaps) in enumerate(
      tqdm(loader, desc ="EVALUATING AT BEAM_SIZE " + str(beam_size))):
    image = image.to(DEVICE) # 1,3,256, 256

    # Encode
    encoder_out = encoder(image) # (1, enc_image_size =14, enc_image_size=14, encoder_dim = 2048)
    enc_image_size = encoder_out.size()[1]
    encoder_dim = encoder_out.size()[3]

    # Flatten encoding 
    encoder_out = encoder_out.view(1, -1, encoder_dim) # (1, 14*14, 2048)
    num_pixels = encoder_out.size()[1]

    # we will treat the prob having batch size = beam_size
    encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim) # (beam_size, 14*14, 2048)

    # Tensor to store previous top beam_size words at each step, now they are just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]* beam_size]).to(DEVICE)

    # Tensor to store top beam_size sequences; now they are just <start>
    seqs = k_prev_words #(beam_size,1)

    # Tensor to store top beam_sequence' scores; noe they are just <start>
    top_k_scores =torch.zeros(beam_size, 1).to(DEVICE) # (beam_size, 1)

    # List to store completed seq and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding 
    step = 1
    h,c = decoder.init_hidden_state(encoder_out)

    # s is a number <= beam_size, beacause sequences are removed from this process once they hit <end>
    while True:
      embeddings = decoder.embedding(k_prev_words).squeeze(1) # (batch_size, embeded_dim)

      awe, _ = decoder.attention(encoder_out, h) # attention_weighted_encoding =(s, encoder_dim=2048), _ = (s, num_pixels=14*14)
      gate = decoder.sigmoid(decoder.f_beta(h)) # gating scaler, (s, encoder_dim =2048)
      awe *= gate
      
      h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

      scores = decoder.fc(h) # (s, vocab_size)
      scores = F.log_softmax(scores, dim=1)
      
      ###### up to here, same as model.py() #####
      
      # Add
      scores = top_k_scores.expand_as(scores) + scores # (s, vocab_size)

      # For first step, all k points will have same scores (since same k previous words, h,c)
      if step == 1:
          top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)  # (s)
      else:
          # Unroll and find top scores, and their unrolled indices
          top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)  # (s)


      #print("\nbefore division, top_k_words: ", top_k_words)
      
      # convert unrolled indices into actual indices of scores
      prev_word_inds = (top_k_words / vocab_size).type(torch.long) # (s)
      next_word_inds = top_k_words % vocab_size # (s)

      # add new word to sequence
      print("\nafter / division, top_k_words: ", prev_word_inds)
      #print("\nafter % division, top_k_words: ",next_word_inds.unsqueeze(1))
      seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

      # Which sequences are incomplete (didn't reach <end>)?
      incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                         next_word != word_map['<end>']]
      complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

      # Set aside complete sequences
      if len(complete_inds) > 0:
        complete_seqs.extend(seqs[complete_inds].tolist())
        complete_seqs_scores.extend(top_k_scores[complete_inds])

      beam_size -= len(complete_inds)

      # proceed with incomplete seq
      if beam_size ==0:
        break
      seqs = seqs[incomplete_inds]
      h = h[prev_word_inds[incomplete_inds]]
      c = c[prev_word_inds[incomplete_inds]]
      encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
      top_k_scores = top_k_scores[prev_word_inds[incomplete_inds]]
      k_prev_words = k_prev_words[prev_word_inds[incomplete_inds]]

      # Break if things have been going too long
      if step > 30:
        break
      step +=1

    ##### outside the while True loop ####
    j = complete_seqs_scores.index(max(complete_seqs_scores)) # find argwhere(max of complete_seq_scores)
    seq = complete_seqs[j]
    if i % 200 ==0:
      # re-arrange image parameters so that we can use imshow(H,W,Channel)
      changed_image = image.view(image.shape[1], image.shape[2], image.shape[0])
      plt.imshow(changed_image)
      print("The predicted caption is: ", seq)

    # References
    img_caps = allcaps[0].tolist()
    img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps))  # remove <start> and pads
    references.append(img_captions)

    # Hypotheses
    hypothesis.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

    assert len(references) == len(hypothesis)

  #### outside the for each img loop ####
  # cauclualte Bleu-4 score
  bleu4 = corpus_bleu(references, hypothesis)

  return bleu4

if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))