from cv2 import sort
import torch
from torch import nn
import torchvision

class Encoder(nn.Module):
    
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        
        self.encoded_image_size = encoded_image_size
        
        # use pretrained ResNet101 as our encoder
        resnet = torchvision.models.resnet101(pretrained=True)
        
        # remove the last two layers (maxpool and dense layer)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # resize the image to fixed size to allow images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.encoded_image_size, self.encoded_image_size))
        
    def forward(self, x):
        x = self.resnet(x) # Bx2048xH/32xW/32
        x = self.adaptive_pool(x) # Bx2048x14x14
        x = x.permute(0, 2, 3, 1) # Bx14x14x2048
        return x
    
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
    
class Decoder(nn.Module):
    
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.2):
        
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim+encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def init_hidden(self, encoder_out):
        
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        
        batch_size, _, _, encoder_dim = encoder_out.size()
        vocab_size = self.vocab_size
        
        # flatten the image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # batch_size, num_pixel, encoder_dim
        num_pixel = encoder_out.size(1)
        
        # sort input data by decreasing caption_length
        caption_lengths, sort_index = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_index]
        encoded_captions = encoded_captions[sort_index]
        
        # embedding
        embeddings = self.embedding(encoded_captions) # batch_size, max_caption_length, embed_dim
        
        # initialize LSTMCell
        h, c = self.init_hidden(encoder_out)
        
        decode_lengths = (caption_lengths - 1).tolist()
        
        # container to store predictions score and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixel).to(device)
        
        # at the time step
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, encoded_captions, decode_lengths, alphas, sort_index
        