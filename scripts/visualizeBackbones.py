# ##
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

#change working directory to root
ROOT_DIR = os.getcwd()
while os.path.basename(ROOT_DIR) != 'VisIrNet':
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR,'..'))
sys.path.insert(0,ROOT_DIR)
os.chdir(ROOT_DIR)

ROOT_DIR = Path(ROOT_DIR)

print(tf.__version__)
devices = tf.config.list_physical_devices('GPU')
print("len(devices): ", len(devices))
print(f"available GPUs: {devices}");


# ##
## gpu setup 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
###

# ##
# config file to load will be passed as an argument
# get run parameters

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', 
                        action = "store", 
                        dest = "config_file",
                        default = "skydata_default_config.json",
                        help = 'specify config file to load')

input_arguments = parser.parse_args()

from Tools.configurations_parser import ConfigurationParser
# load configurations
configs = ConfigurationParser.getConfigurations(configs_path = 'configs', 
                                                config_file = str(input_arguments.config_file))


# print configurations
# ConfigurationParser.printConfigurations(configs)


# ## [markdown]
# **Dataloaders**

# ##
import data_setup

import model_setup
import Utils


import Tools.utilities as common_utils
# ##
import Tools.loss_functions as loss_functions
import Tools.datasetTools as DatasetTools


def visualize_and_save(samples_to_visualize=10,
                       model = None,
                       dataloader = None,
                       loss_functions_used = None,
                       save_path = None,
                       dataset_name = None,
                       ):
        
    # ##
    for batch in tqdm(dataloader.take(samples_to_visualize)):
        input_images, template_images, labels,_instances = batch
        
        gt_matrix = DatasetTools.get_ground_truth_homographies(labels)
        warped_inputs, _ = DatasetTools._get_warped_sampled(images = input_images,  homography_matrices = gt_matrix)
        
        rgb_fmaps , ir_fmaps = model.call((input_images, template_images), training=False)
        
        
        warped_fmaps,_ = DatasetTools._get_warped_sampled( images = rgb_fmaps, 
                                                        homography_matrices = gt_matrix)


        total_loss , detailed_batch_losses = loss_functions.get_losses_febackbone( warped_inputs,
                                                                                    template_images,
                                                                                    warped_fmaps,
                                                                                    ir_fmaps,
                                                                                    loss_functions_used)



        all_data_to_plot = {
                            0:input_images,
                            1:template_images,
                            2:warped_inputs,
                            4:rgb_fmaps,
                            5:ir_fmaps,
                            6:warped_fmaps,
                                        }
        
        common_utils.plot_showcase_and_save_backbone_results( all_data_to_plot,
                                                _instances,
                                                save_path = save_path if save_path else "resources/backbone-showcase",
                                                dataset_name = configs.dataset,
                                                loss_functions_used = loss_functions_used
                                                )
        



### setup dataloaders 
train_dataloader,test_dataloader = data_setup.create_dataloaders(dataset=configs.dataset, 
                                                                BATCH_SIZE=1,
                                                                SHUFFLE_BUFFER_SIZE=configs.SHUFFLE_BUFFER_SIZE
                                                                )

len(train_dataloader), len(test_dataloader)


### vis loop 

samples_to_visualize = 10
# for loss_function_used in ["mse_pixel","ssim_pixel","mae_pixel","sse_pixel"]:
for loss_function_used in ["sse_pixel"]:
    print(f"[INFO] visualizing {loss_function_used} ...")
    ### setup model
    # from_checkpoint = "SkyData-sse_pixel-featureEmbeddingBackbone-1-50-c139e1fb63f240789439f2bfc7cba603-1.keras"
    from_checkpoint = "latest"
    save_path = f"models/{configs.dataset}/{loss_function_used}"

    pattern = f"*{loss_function_used}-featureEmbeddingBackbone*" if str(from_checkpoint)=="latest" else f"*{from_checkpoint}*"

    try:
        model_name = common_utils.latest_file(Path(save_path), pattern=pattern)
        model = common_utils.load_model(model_name)
    except Exception as e:
        print(e)
        print("error loading model")
        exit()
        
    ### visualize and save
    visualize_and_save(samples_to_visualize = samples_to_visualize,
                        model = model,
                        dataloader = test_dataloader,
                        loss_functions_used = loss_function_used,
                        save_path = None,
                        dataset_name = configs.dataset,
                        )
    print(f"[INFO] visualizing {loss_function_used} done")






