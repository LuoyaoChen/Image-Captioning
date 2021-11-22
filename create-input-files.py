from utils import *



JSON_PATH = './data/annotations/dataset_coco.json'
IMG_PATH = './data/image/'
MAX_LEN = 100
MIN_WORD_FREQ = 5
CAPTION_PER_IMAGE = 5

create_input_files(dataset='coco', karpathy_json_path = JSON_PATH, image_folder=IMG_PATH, 
                   captions_per_image= CAPTION_PER_IMAGE, 
                   min_word_freq=MIN_WORD_FREQ, output_folder='./json_folder/',
                   max_len=MAX_LEN)