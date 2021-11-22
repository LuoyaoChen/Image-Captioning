import time
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *

# Data parameters
DATA_FOLDER = './json_folder/'
DATA_NAME = 'coco_5_cap_per_img_5_min_word_freq'

# Model parameters
EMD_DIM = 512 # dim of word embeddings
ATTENTION_DIM = 512 # dimension of attention (linear layers)
DECODER_DIM = 512 # dimension of LSTM
DROPOUT = 0.5 

DEVICE = torch.device('cuda',0) # set device for mode and Pytorch tensors
cudnn.benchmark = True # True only if inputs to model are fixed size. Otherwise a log of computational overhead

# Traning parameters
START_EPOCH = 6
EPOCHS = 20 # number of epochs to train for, when early stopping is not triggered
EPOCH_SINCE_IMPROVEMENT = 0 # keep track of number of epochs since there has been improvement in validation BLEU
BATCH_SIZE = 80

WORKERS = 1 # for data loading. Right now, only 1 works for h5py
ENCODER_LR = 1e-4 # lr for encodder fine-tuning
DECODER_LR = 4e-4 # lr for decoder
GRAD_CLIP = 5.  # clip gradients at an absolute value of 
ALPHA_C = 1. # regularization parameter for 'doubly stochatic attention'as in the paper
BEST_BLEU4 = 0. # bleu score right now
PRINT_FREQ = 100 # print training/validation stats every _ batches
FINE_TUNE_ENCODER = False # whether we fine-tune encoder?
CHECKPOINT = './checkpoints/checkpt_7_epoch.pt' # path to checkpoint, None if none

def main():
  '''
    training and validation
  '''
  global BEST_BLEU4, EPOCH_SINCE_IMPROVEMENT, CHECKPOINT, START_EPOCH, FINE_TUNE_ENCODER, DATA_NAME, WORD_MAP
  
  # read word map
  word_map_file = os.path.join(DATA_FOLDER, 'WORDMAP_' + DATA_NAME + '.json')
  with open(word_map_file, 'r') as j:
    WORD_MAP = json.load(j)

  # initialize/ load checkpoint
  if CHECKPOINT is None:
    decoder = DecoderWithAttention(attention_dim = ATTENTION_DIM,
                                   embed_dim = EMD_DIM, 
                                   decoder_dim = DECODER_DIM,
                                   vocab_size = len(WORD_MAP),
                                   encoder_dim = 2048,
                                   dropout = DROPOUT)
    decoder_optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr = DECODER_LR)
    encoder = Encoder()
    encoder.fine_tune(FINE_TUNE_ENCODER) # FINE_TUNE_ENCODER = False
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=ENCODER_LR) if FINE_TUNE_ENCODER else None
  
  else:
    CHECKPOINT = torch.load(CHECKPOINT)
    start_epcoh = CHECKPOINT['epoch'] + 1
    EPOCH_SINCE_IMPROVEMENT = CHECKPOINT['epochs_since_improvement']
    best_bleu4 = CHECKPOINT['bleu-4']
    decoder = CHECKPOINT['decoder']
    decoder_optimizer = CHECKPOINT['decoder_optimizer']
    encoder = CHECKPOINT['encoder']
    encoder_optimizer = CHECKPOINT['encoder_optimizer']
    if FINE_TUNE_ENCODER is True and encoder_optimizer is None:
      encoder.fine_tune(FINE_TUNE_ENCODER)
      encoder_optimizer = torch.optim. Adam(params = filter(lambda p: p.required_grad, decoder.parameters()),
                                            lr = DECODER_LR)
      
  # move to GPU, if possible
  decoder = decoder.to(DEVICE)
  encoder = encoder.to(DEVICE)

  # Loss function
  criterion = nn.CrossEntropyLoss().to(DEVICE)

  # Custom DataLoaders
  normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
  train_loader = torch.utils.data.DataLoader(
     Caption_Dataset(DATA_FOLDER, DATA_NAME, 'TRAIN', state = 'TRAIN', transform = transforms.Compose([normalize])),
     batch_size=BATCH_SIZE, shuffle= True, num_workers = WORKERS, pin_memory = True
  )
  val_loader = torch.utils.data.DataLoader(
     Caption_Dataset(DATA_FOLDER, DATA_NAME, 'VAL',state = 'VAL', transform = transforms.Compose([normalize])),
     batch_size=BATCH_SIZE, shuffle= True, num_workers = WORKERS, pin_memory = True
  )
  
  # Epochs
  for epoch in range(START_EPOCH, EPOCHS):
    # decay en/decoder's lr if there is no improvement for 8 consecutive epochs; terminate training after 20 epochs
    if EPOCH_SINCE_IMPROVEMENT ==20:
      break
    
    if EPOCH_SINCE_IMPROVEMENT > 0 and EPOCH_SINCE_IMPROVEMENT %8==0:
      adjust_learning_rate(decoder_optimizer, 0.8)
      if FINE_TUNE_ENCODER:
        adjust_learning_rate(encoder, 0.8)


    # one epoch training using train() defined below
    train(train_loader = train_loader,
          encoder =  encoder,
          decoder = decoder, 
          criterion = criterion,
          encoder_optimizer = encoder_optimizer,
          decoder_optimizer= decoder_optimizer,
          epoch = epoch)
    
    # one epoch validation using validate() below
    recent_bleu4 = validate(val_loader=val_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion)
      # check if there is improvement
    is_best = recent_bleu4 > BEST_BLEU4 
    BEST_BLEU4 = max(recent_bleu4, BEST_BLEU4)

    if not is_best:
      EPOCH_SINCE_IMPROVEMENT+=1
      print("\nEpochs since last improvement: %d\n" %(EPOCH_SINCE_IMPROVEMENT,))
    else:
      EPOCH_SINCE_IMPROVEMENT = 0

    # save checkpoint
    save_checkpoint(epoch, EPOCH_SINCE_IMPROVEMENT, encoder, decoder, 
                    encoder_optimizer, decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
  '''
    perform 1 epoch training
    - train_loader: DataLoader for training data
    - encoder: encoder model
    - decoder: decoder model
    - cirterion: loss
    - encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    - decoder_optimzer: optimizer to update decoder's weights 
    - epoch: epoch number
  '''
  encoder.train()
  decoder.train()

  batch_time = AverageMeter() # forward pro +backward prop time. note AverageMeter() is defined in utils.py
  data_time = AverageMeter() # data loading time
  losses = AverageMeter() # loss (per word decoded)
  top5accs = AverageMeter() # top 5 accuracies

  start = time.time()

  # Batches
  for i, (imgs, caps, caplens) in enumerate(train_loader):
    data_time.update(time.time()- start)

    # Move to GPU
    imgs = imgs.to(DEVICE)
    caps = caps.to(DEVICE)
    caplens = caplens.to(DEVICE)

    # forward 
    out = encoder(imgs)
    out, caps_sorted, decode_lengths, alphas, sort_ind = decoder(out, caps, caplens)

    # Since we decOded starting with <start>, the target are words after <start>, up to <end>
    targets = caps_sorted[:,1:]

    # Remove pads and those we didn't decode. This can be done using "pack_padded_sequence", here pack means "compact(压紧）"
    out = pack_padded_sequence(out, decode_lengths, batch_first=True)[0]
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
    
    # Calculate loss
    loss = criterion(out, targets)

    # Doubly stochastic attention regularization
    loss += ALPHA_C * ((1. - alphas.sum(dim=1))**2).mean()

    # backward
    decoder_optimizer.zero_grad()
    if encoder_optimizer is not None:
      encoder_optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    if GRAD_CLIP is not None:
      clip_gradient(decoder_optimizer, GRAD_CLIP)
      if encoder_optimizer is not None:
        clip_gradient(encoder_optimizer, GRAD_CLIP)

    # update weights
    decoder_optimizer.step()
    if encoder_optimizer is not None:
      encoder_optimizer.step()

    # keep track of metrics
    losses.update(loss.item(), sum(decode_lengths)) # uses AverageMeter class in utils.py
    top5accs.update(accuracy(out, targets, 5) , sum(decode_lengths))
    batch_time.update(time.time() - start)

    start = time.time()

    #print status
    if i % PRINT_FREQ == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses,
                                                                    top5=top5accs)
            )
    


def validate(val_loader,encoder, decoder, criterion):
  '''
    perform one epoch's validtaion
     -- val_loader: Dataloder for validation data
     -- encoder : encoder model
     -- decoder: decoder model
     -- cirterion: loss layer
     return: Bleu-4 socre
  '''
  decoder.eval()
  if encoder is not None:
    encoder.eval()

  batch_time = AverageMeter()
  losses = AverageMeter()
  top5accs = AverageMeter()

  start = time.time()

  references = list() # true captions
  hypothesis = list() # hypothesis captions

  with torch.no_grad():
    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
      # Move to device, if available
      imgs = imgs.to(DEVICE)
      caps = caps.to(DEVICE)
      caplens = caplens.to(DEVICE)
    
      # forward
      if encoder is not None:
        out = encoder(imgs)

        out, caps_sorted, decode_lengths, alphas, sort_ind = decoder(out, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        ## print(targets.shape) #size = torch.Size(32,101)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        out_copy = out.clone()
        out = pack_padded_sequence(out, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # loss
        loss = criterion(out, targets)

        # doubly stochastic attention reugularization
        loss += ALPHA_C * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(out, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % PRINT_FREQ == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

        # Bleu score
        # score references (true captions), and hypothesis for each image
        # If for n images, we have n hypothesis, and reference a,b,c.. for each images, we need --
        # reference = [[ref1a,ref1b,ref1c], [red2a, ref2b, ref2c], ...], hypothesis [hyp1, hyp2, ...]

        # Reference
        allcaps = allcaps[sort_ind]
        for j in range(allcaps.shape[0]):
          img_caps = allcaps[j].tolist()
          img_captions = list(
              map(lambda c: [w for w in c if w not in {WORD_MAP['<start>'], WORD_MAP['<pad>']}],
                  img_caps) # remove <start> and <pad>
          )
          references.append(img_captions)

        # Hypothesis
        _, preds = torch.max(out_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
          temp_preds.append(preds[j][:decode_lengths[j]]) # remove pads
        preds = temp_preds
        hypothesis.extend(preds)

        assert len(references) ==len(hypothesis)
      
      # BlEU score:
      bleu4 = corpus_bleu(references, hypothesis)
      if i % PRINT_FREQ == 0:
        print(
            '* LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg: .3f}, BLEU-4 - {bleu}'.format(
                loss = losses,
                top5=top5accs,
                bleu = bleu4
            )
        )
  return bleu4


if __name__ == "__main__":
  main()
  
