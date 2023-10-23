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

train_dataloader,test_dataloader = data_setup.create_dataloaders(dataset=configs.dataset, 
                                                                BATCH_SIZE=1,
                                                                SHUFFLE_BUFFER_SIZE=configs.SHUFFLE_BUFFER_SIZE
                                                                )

len(train_dataloader), len(test_dataloader)
#

# ## [markdown]
# ## **Model**

# ##
## model name to load
import model_setup
import Utils


import Tools.utilities as common_utils

loss_functions_used ="ssim_pixel"
from_checkpoint = "SkyData-ssim_pixel-featureEmbeddingBackbone-1-50-0ca7a287425746488530f7ee0e517ab7-5.keras"
save_path = f"models/{configs.dataset}"

pattern = f"*{loss_functions_used}-featureEmbeddingBackbone*" if str(from_checkpoint)=="latest" else f"*{from_checkpoint}*"

try:
    model_name = common_utils.latest_file(Path(save_path), pattern=pattern)
    model = common_utils.load_model(model_name)
except Exception as e:
    print(e)
    print("error loading model")
    exit()

# ##

## 

def plot_showcase_and_save_backbone_results( all_data_to_plot,
                                        _instances,
                                        save_path = "resources/backbone-showcase",
                                        dataset_name = "SkyData",
                                        loss_functions_used = "mse_pixel"
                                        ):

    

    # input_images = all_data_to_plot[0]
    template_images = all_data_to_plot[1]
    warped_inputs = all_data_to_plot[2]
    # rgb_fmaps = all_data_to_plot[4]
    ir_fmaps = all_data_to_plot[5]
    warped_fmaps = all_data_to_plot[6]

    for i_th in range(len(all_data_to_plot[0])):
        data_to_plot = {k:np.array(v[i_th]).clip(0,1) for k,v in all_data_to_plot.items()}
        
        summed_data = {
                3: 0.9 * warped_inputs[i_th] + .2 * warped_fmaps[i_th],
                7: 0.05 *template_images[i_th] + 1 *  tf.cast(ir_fmaps[i_th],tf.float32)
        }
        data_to_plot.update(summed_data)
        
        fig, axs = plt.subplots(2, 4, figsize=(8, 5), constrained_layout=True)
        axs = axs.ravel()

        # fig = plt.figure(figsize=(20, 20))
        # axs = fig.subplots(3, 2)
        titles=["input_images","template_images","warped_inputs","summed_rgb","rgb_fmaps","ir_fmaps","warped_fmaps","summed_ir"]
        for i, ax in enumerate(axs):
            ax.axis('off')
            ax.set_title(titles[i])
            


        for i, data_i in data_to_plot.items():
            axs[i].imshow(np.array(data_i).clip(0,1))
        # fig.tight_layout()
        # plt.show()

        #saving showcase
        
        save_path = Path(f"{save_path}/{dataset_name}/{loss_functions_used}")
        save_name = f"{_instances.numpy()[i_th].decode('utf-8')}.png"
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_as= str(save_path/save_name)
        
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
        plt.close()
            

# ## [markdown]
# ## **Forward Pass**
# 

# ##
samples_to_visualize = 20

# ##
import Tools.loss_functions as loss_functions
import Tools.datasetTools as DatasetTools

# ##
for batch in train_dataloader.take(samples_to_visualize):
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
    
    plot_showcase_and_save_backbone_results( all_data_to_plot,
                                            _instances,
                                            save_path = "resources/backbone-showcase",
                                            dataset_name = configs.dataset,
                                            loss_functions_used = loss_functions_used
                                            )
    




# ##



