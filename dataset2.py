import os, sys
import json, h5py
import torch
from torch.utils.data import Dataset

class Caption_Dataset(Dataset):
    
    def __init__ (self, data_folder, split):
        self.split = split
        assert self.split in {'TRAIN', 'VALID', 'TEST'}
        
        # open hdf5 file
        self.h = h5py.File(os.path.join('IMAGES_COCO_' + split + '.hdf5'), 'r')
        self.imgs = self.h['images']
        
        # caption per image
        self.cpi = 5
        
        # load encoded captions
        with open(os.path.join('CAPTIONS_COCO_' + split + '.json'), 'r') as j:
            self.captions = json.load(j)
            
        with open(os.path.join('CAPLENS_COCO_' + split + '.json'), 'r') as j:
            self.caplens = json.load(j)
            
        self.dataset_size = len(self.captions)
        
    def __getitem__(self, i):
        img = torch.Float(self.imgs[i // self.cpi] / 255.0)
        
        caption = torch.LongTensor(self.captions[i])
        
        caplen = torch.LongTensor(self.caplens[i])
        
        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # for validation or testing, also return all 'captions_per_image' to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions
        
    def __len__(self):
        return self.dataset_size