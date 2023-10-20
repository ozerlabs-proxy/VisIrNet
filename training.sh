#!bash

# This script is used to train the model.


#### ["mse_pixel", "mae_pixel", "sse_pixel", "ssim_pixel"]
#### ["l1_homography_loss" , "l2_homography_loss" , "l1_corners_loss" , "l2_corners_loss"]

# conda activate VisIrNet

nvidia-smi && python Train.py --config-file skydata_default_config.json --b_loss_function mse_pixel  --train_first_stage True

