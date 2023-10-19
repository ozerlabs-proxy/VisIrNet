#!bash

# This script is used to train the model.


#### ["mse_pixel", "mae_pixel", "sse_pixel", "ssim_pixel"]
#### ["l1_homography_loss" , "l2_homography_loss" , "l1_corners_loss" , "l2_corners_loss"]

# conda activate VisIrNet

###################### SKYDATA  ############################
##>>>>>>>>>>>>>>>>>>>>>>>> skydata backbone <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
python Train.py --config-file skydata_default_config.json --b_loss_function ssim_pixel --train_first_stage True 
# python Train.py --config-file skydata_default_config.json --b_loss_function mse_pixel  --train_first_stage True
# python Train.py --config-file skydata_default_config.json --b_loss_function mae_pixel  --train_first_stage True
# python Train.py --config-file skydata_default_config.json --b_loss_function sse_pixel  --train_first_stage True

##>>>>>>>>>>>>>>>>>>>>>>>> regression head  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# python Train.py --config-file skydata_default_config.json --b_loss_function ssim_pixel   --train_second_stage True --r_loss_function l2_homography_loss
# python Train.py --config-file skydata_default_config.json --b_loss_function mse_pixel  --train_second_stage True --r_loss_function l2_homography_loss
# python Train.py --config-file skydata_default_config.json --b_loss_function mae_pixel  --train_second_stage True --r_loss_function l2_homography_loss
# python Train.py --config-file skydata_default_config.json --b_loss_function sse_pixel  --train_second_stage True --r_loss_function l2_homography_loss



# ###################### VEDAI  ############################
# ##>>>>>>>>>>>>>>>>>>>>>>>> veai backbone <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# python Train.py --config-file vedai_default_config.json --b_loss_function ssim_pixel --train_first_stage True 
# # python Train.py --config-file vedai_default_config.json --b_loss_function mse_pixel  --train_first_stage True
# # python Train.py --config-file vedai_default_config.json --b_loss_function mae_pixel  --train_first_stage True
# # python Train.py --config-file vedai_default_config.json --b_loss_function sse_pixel  --train_first_stage True

# ##>>>>>>>>>>>>>>>>>>>>>>>> regression head  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# python Train.py --config-file vedai_default_config.json --b_loss_function ssim_pixel   --train_second_stage True --r_loss_function l2_homography_loss
# # python Train.py --config-file vedai_default_config.json --b_loss_function mse_pixel  --train_second_stage True --r_loss_function l2_homography_loss
# # python Train.py --config-file vedai_default_config.json --b_loss_function mae_pixel  --train_second_stage True --r_loss_function l2_homography_loss
# # python Train.py --config-file vedai_default_config.json --b_loss_function sse_pixel  --train_second_stage True --r_loss_function l2_homography_loss


