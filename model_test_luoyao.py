import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

global DEVICE 
DEVICE = torch.device('cuda', 0) # set device for mode and Pytorch tensors


word_map_file = os.path.join('json_folder', 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    Caption_Dataset('json_folder', 'coco_5_cap_per_img_5_min_word_freq', 'TRAIN', state = 'TRAIN', transform=transforms.Compose([normalize])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True)


decoder = DecoderWithAttention(DEVICE, 
                               attention_dim=512,
                              embed_dim=512,
                              decoder_dim=512,
                              vocab_size=len(word_map),
                              encoder_dim = 2048,
                              dropout=0.5).to(DEVICE)

encoder = Encoder().to(DEVICE)

# for i, (imgs, caps, caplens) in enumerate(train_loader):

#     # Move to GPU, if available
#     imgs = imgs.to(DEVICE)
#     caps = caps.to(DEVICE)
#     caplens = caplens.to(DEVICE)

#     # Forward prop.
#     imgs = encoder(imgs)
#     print(imgs.shape) #??
#     scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
#     print(scores.shape, caps_sorted.shape, decode_lengths.shape, alphas.shape, sort_ind.shape) # ??
    
    
    
data=next(iter(train_loader))
# print()
for x in data:
    print(x.size())
imgs, caps, caplens = data
# print(imgs.size()) # 1, 3, 256, 256
# print(caps.size()) # 1, 102
# print(caplens.size()) # 1, 1
# Move to GPU
imgs = imgs.to(DEVICE)
caps = caps.to(DEVICE)
caplens = caplens.to(DEVICE)

# forward 
# encode
out = encoder(imgs)
print(out.size()) # (1,14,14,2048)
out1 = out.reshape(1,-1,2048)
print(out1.size())  # ([1, 196, 2048])
out2 = out1.expand(3, 196, 2048)
print(out2.size()) # ([3, 196, 2048])
print(out2.mean(dim=1).size()) # ([3, 2048])

# decode
scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(out, caps, caplens)
print(scores.size())  # ([1, 8.., 6504])
print(caps_sorted.size()) # ([1, 102])
print(decode_lengths)  # [11..]
print(alphas.size()) #  ([1, 12, 196])
print(sort_ind.size()) # [1]

h,c = decoder.init_hidden_state(out2)
k_prev_words = torch.LongTensor([word_map['<start>']]*3).to(DEVICE)
seqs = k_prev_words 
#while True:
embeddings = decoder.embedding(k_prev_words) #(s, embded_dim =512)
print("embedding size is", embeddings) # (3, 512)
awe, alpha = decoder.attention(out2, h)
print(alpha.size()) # [3, 196]


h,c = decoder.decode_step(torch.cat([embeddings, awe], dim=1),(h,c)) # (s=3, decoder_dim = 2048)

scores = decoder.fc(h)
print(f"scores.size() is :{scores.size()}")

# Add
# scores = top_k_scores.expand_ax(scores) + scores