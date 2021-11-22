import torch
from torch.utils.data import Dataset
import h5py
import json
import os

class Caption_Dataset(Dataset):
  '''
  A Pytorch Dataset class to be used in a Pytorch DataLoader to create batches
  '''

  def __init__(self, data_folder, data_name, split, state, transform = None):
    '''
    data_folder: folder where json files are stored
    data_name: base name of processed datasets
    split: 'TRAIN', 'TEST', 'VAL'
    state: the actual state: 'TRAIN', 'VAL', 'TEST'
    transform: data transformation
    '''

    self.split = split
    assert self.split in {'TRAIN', 'VAL', 'TEST'}
    self.state= state 
    
    # open hdf5 file where images are stored
    self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
    self.imgs = self.h['images']

    # Captions per image
    self.cpi= self.h.attrs['captions_per_image']

    # Load encoded captions (completely into memory)
    with open(os.path.join(data_folder, self.split + '_CAPTIONS_'+ data_name + '.json'), 'r') as j:
      self.captions = json.load(j)

    # Load caption length (completely into memory)
    with open(os.path.join(data_folder, self.split + '_CAPLENS_'+ data_name + '.json'), 'r') as j:
      self.caplens = json.load(j)

    # Pytorch Transforation
    self.transform = transform 

    # Total number of datapoints
    self.dataset_size = len(self.captions)

  def __getitem__(self, i):
    # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
    img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
    if self.transform is not None:
        img = self.transform(img)

    caption = torch.LongTensor(self.captions[i])

    caplen = torch.LongTensor([self.caplens[i]])

    if self.state == 'TRAIN':
        return img, caption, caplen
    else:
        # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        all_captions = torch.LongTensor(
            self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        return img, caption, caplen, all_captions

  def __len__(self):
    return self.dataset_size

     