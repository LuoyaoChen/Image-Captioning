import os, sys
import json
import random
import numpy as np
import h5py
from tqdm import tqdm
import cv2
from collections import Counter

'''
captions_train2014.json
info: COCO 2014 Dataset (http://cocodataset.org) version 1.0
images: {license, filename, coco_url, height, width, id}
annotations: {image_id, id, caption}
'''

"""
Karpathy's Split: dataset_coco.json
{images}: [sentences] (tokens) [split] (train, test, val)
"""

JSON_PATH = './data/annotations/dataset_coco.json'
IMG_PATH = './data/image/'
MAX_LEN = 100
MIN_WORD_FREQ = 5
CAPTION_PER_IMAGE = 5
SAVE = False
dataset = 'coco'
output_folder = './tmp/'

def main():
    # initialize
    word_counter = Counter()
    train_image_paths = []
    train_image_captions = []
    test_image_paths = []
    test_image_captions = []
    valid_image_paths = []
    valid_image_captions = []
    
    # load json file
    with open(JSON_PATH, 'r') as json_file:
        data = json.load(json_file)
        #print(data)

    # loop through all images in json file
    for image in data['images']:
        captions = []
        # use token in json file to update word counter
        for sentence in image['sentences']:
            word_counter.update(sentence['tokens'])
            # save caption in the same time
            if len(sentence['tokens']) <= MAX_LEN:
                captions.append(sentence['tokens'])

        # generate path
        path = os.path.abspath(os.path.join(IMG_PATH, image['filepath'], image['filename']))

        # separate into train/valid/test sets
        if image['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif image['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
        elif image['split'] in {'val'}:
            valid_image_paths.append(path)
            valid_image_captions.append(captions)

    # create word map
    words = [w for w in word_counter.keys() if word_counter[w] > MIN_WORD_FREQ]
    word_map = {word: index+1 for index, word in enumerate(words)}
    # add special words
    word_map['<unknown>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(CAPTION_PER_IMAGE) + '_cap_per_img_' + str(MIN_WORD_FREQ) + 'MIN_WORD_FREQ'

    # Save word map to a JSON\
    if SAVE:
        with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
            json.dump(word_map, j)

    # save data to HDF5 file
    for image_path, image_caption, split in [(train_image_paths, train_image_captions, 'TRAIN'), (valid_image_paths, valid_image_captions, 'VALID'), (test_image_paths, test_image_captions, 'TEST')]:
        
        with h5py.File(os.path.join('IMAGES_COCO_' + split + '.hdf5'), 'w') as h:
            # create dataset for storing images
            images = h.create_dataset('images', (len(image_path), 3, 256, 256), dtype='uint8')
            print(f'\n Saving {split} images and captions ...')
            
            encoded_captions = []
            caption_lens = []
            for index, path in enumerate(tqdm(image_path)):
                # sample captions from dataset
                if len(image_caption[index]) < CAPTION_PER_IMAGE:
                    captions = image_caption[index] + [random.choice(image_caption[index]) for _ in range(CAPTION_PER_IMAGE - len(image_caption[index]))]
                else:
                    captions = random.sample(image_caption[index], k=CAPTION_PER_IMAGE)
                
                # sanity check
                assert len(captions) == CAPTION_PER_IMAGE
                
                # read images
                img = cv2.imread(image_path[index], cv2.IMREAD_COLOR) / 255.0
                img = cv2.resize(img, (256, 256))
                img = np.swapaxes(img, 0, 2)

                assert img.shape == (3, 256, 256)
                
                # save image to HDF5
                images[index] = img
                
                # encode captions
                for cap in captions:
                    encoded_cap = [word_map['<start>']] + [word_map.get(word, word_map['<unknown>']) for word in cap] + \
                        [word_map['<end>']] + [word_map['<pad>']] * (MAX_LEN - len(cap))
                        
                    cap_len = len(cap) + 2
                    
                    encoded_captions.append(encoded_cap)
                    print(len(encoded_cap))
                    caption_lens.append(cap_len)
                    
            # sanity check
            assert images.shape[0] * CAPTION_PER_IMAGE == len(encoded_captions) == len(caption_lens)
            
            if SAVE:
                # save encoded caption and caption length
                with open(os.path.join('CAPTIONS_COCO_' + split + '.json'), 'w') as j:
                    json.dump(encoded_captions, j)
                
                with open(os.path.join('CAPLENS_COCO_' + split + '.json'), 'w') as j:
                    json.dump(caption_lens, j)
                                    
                
if __name__ == '__main__':
    # change the working directory to the location of this script
    os.chdir(sys.path[0])

    main()