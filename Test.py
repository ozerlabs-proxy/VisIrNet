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

# tf.keras.mixed_precision.set_global_policy('float64')
tf.keras.backend.set_floatx('float64')
print(f"[INFO] using {tf.keras.backend.floatx()} as default float type")

#change working directory to root
ROOT_DIR = os.getcwd()
while os.path.basename(ROOT_DIR) != 'VisIrNet':
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR,'..'))
sys.path.insert(0,ROOT_DIR)
os.chdir(ROOT_DIR)

ROOT_DIR = Path(ROOT_DIR)
tf.config.set_visible_devices([], 'GPU')
print(tf.__version__)
devices = tf.config.list_physical_devices('GPU')
print("len(devices): ", len(devices))
print(f"available GPUs: {devices}");


## tf 
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





# ## [markdown]
# **Configurations**

# ##
# config file to load will be passed as an argument
# get run parameters

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--config-file', action = "store",  dest = "config_file", default = "default_config.json",help = 'specify config file to load')
parser.add_argument('--b_loss_function', action = "store",  dest = "b_loss_function", default = None ,help = 'f embedding block loss')
parser.add_argument('--r_loss_function', action = "store",  dest = "r_loss_function", default = None ,help = 'regression head  loss')
parser.add_argument('--train_first_stage', action = "store",  dest = "train_first_stage", default = None ,help = 'train first stage')
parser.add_argument('--train_second_stage', action = "store",  dest = "train_second_stage", default = None ,help = 'train second stage')
parser.add_argument('--predicting_homography', action = "store",  dest = "predicting_homography", default = None ,help = 'predicting homography')
input_arguments = parser.parse_args()

from Tools.configurations_parser import ConfigurationParser
# load configurations
configs = ConfigurationParser.getConfigurations(configs_path = 'configs', 
                                                config_file = str(input_arguments.config_file))


## if loss functions are passed as arguments, override the ones in the config file
if input_arguments.b_loss_function is not None:
    configs = configs._replace(B_LOSS_FUNCTION = str(input_arguments.b_loss_function))
if input_arguments.r_loss_function is not None:
    configs = configs._replace(R_LOSS_FUNCTION = str(input_arguments.r_loss_function))
if input_arguments.train_first_stage is not None:
    configs = configs._replace(TrainFirstStage = bool(input_arguments.train_first_stage))
if input_arguments.train_second_stage is not None:
    configs = configs._replace(TrainSecondStage = bool(input_arguments.train_second_stage))    
if input_arguments.predicting_homography is not None:
    configs = configs._replace(R_predicting_homography = bool(input_arguments.predicting_homography))   
    

# print configurations
ConfigurationParser.printConfigurations(configs)



import data_setup

test_dataloader = data_setup.create_dataloader_test(dataset = configs.dataset, 
                                                                BATCH_SIZE = 1
                                                                )

print(f" {len(test_dataloader)}")
#

# ##
import model_setup
import Utils

import engine 


print("*"*25, f"Running Predictions", "*"*25)


from timeit import default_timer as timer
start_time = timer()
# Train model 

# prediction_results = engine.run_predictions(model = regressionHead,
prediction_results = engine.run_predictions(featureEmbeddingBackBone = configs.R_featureEmbeddingBackBone,
                                                test_dataloader = test_dataloader,
                                                dataset_name = configs.dataset,
                                                from_checkpoint = "latest",
                                                save_path = configs.R_save_path,
                                                predicting_homography = configs.R_predicting_homography,
                                                backbone_loss_function = configs.B_LOSS_FUNCTION,
                                                loss_function_to_use = configs.R_LOSS_FUNCTION
                                                )
# End the timer and print out how long it took
end_time = timer()
training_time = (end_time-start_time)/3600
print(f"Total prediction time : {training_time:.2f} hours")

