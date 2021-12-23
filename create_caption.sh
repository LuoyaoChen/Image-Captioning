python caption2.py \
--img='./results_report/inference_images/ResNet50/img_22000_bs_3.png' \
--model='./checkpoints/ResNet152/ResNet152_FineTune_E16_BEST_ResNet152_FineTune_E16_checkpt_7_epoch_dev.pt' \
--word_map='./json_folder/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' \
--beam_size=3 \
--dont_smooth