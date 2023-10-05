# ## 
import os
import sys
from pathlib import Path
#change working directory to root
ROOT_DIR = os.getcwd()
while os.path.basename(ROOT_DIR) != 'VisIrNet':
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR,'..'))
sys.path.insert(0,ROOT_DIR)
os.chdir(ROOT_DIR)

ROOT_DIR = Path(ROOT_DIR)


# ## 

import json
import cv2
import tensorflow as tf
import numpy as np
import PIL.Image as Image

import Tools.datasetTools as DatasetTools
import Tools.utilities as common_utils
from tqdm.auto import tqdm

print(tf.__version__)
devices = tf.config.list_physical_devices('GPU')
print("len(devices): ", len(devices))
print(f"available GPUs: {devices}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ##  [markdown]
# **Dataloaders**

# ## 
import data_setup


# try to import the dataset
dataset="SkyData"
BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 100

print("**dataset: ", dataset)
train_dataloader,test_dataloader = data_setup.create_dataloaders(dataset=dataset, 
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                SHUFFLE_BUFFER_SIZE=100
                                                                )
#

# ##  [markdown]
# **Dataset integrity**
# - display samples count
# - samples(images) don't contain NaNs
# - inverse of homography matrix exits
# 

# ## 
dataloaders = {
    "train":train_dataloader,
    "test":test_dataloader
}

dataset_info={}
dataset_info["dataset"]= dataset


# ## 
for split, dataloader in dataloaders.items():
    dataset_info[split]={}
    dataset_info[split]["size"] = len(dataloader)
    dataset_info[split]["NaNs"] = []
    dataset_info[split]["transformed_have_NaNs"] = []
    dataset_info[split]["nonInvertibles"] = []
    dataset_info[split]["input_image_size"] = str(list(dataloader.take(1))[0][0].shape)
    dataset_info[split]["template_image_size"] = str(list(dataloader.take(1))[0][1].shape)
    
    for batch in tqdm(dataloader):
        input_images, template_images, labels ,_instances = batch
        
        # check for nans
        if np.isnan(input_images).any() or np.isnan(template_images).any():
            dataset_info[split]["NaNs"].append(str(_instances.numpy()[0]))
            
        homography_matrices = DatasetTools.get_ground_truth_homographies(labels) 
        inverse_homography_matrices, _is_invertibles = DatasetTools.is_invertible(homography_matrices)   
        
        #invertibility check
        if not _is_invertibles:
            dataset_info[split]["nonInvertibles"].append(str(_instances.numpy()[0]))
            
        warped_images, _transformed_have_nans = DatasetTools._transformed_images(input_images, 
                                                                                homography_matrices)
        # check for nans in transformed images
        if _transformed_have_nans:
            dataset_info[split]["transformed_have_NaNs"].append(str(_instances.numpy()[0]))
            
#save dataset info
common_utils.save_json(dataset_info, 
                        f"resources/datasets_logs/",
                        f"{dataset}_dataset_info.json"
                        )


