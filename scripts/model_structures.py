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


import model_setup
import Utils
from datetime import date



featureEmbeddingBackBone = model_setup.getFeatureEmbeddingBackBone(rgb_inputs_shape=configs.RGB_INPUTS_SHAPE,
                                                        ir_inputs_shape=configs.IR_INPUTS_SHAPE,
                                                        output_channels_per_block=configs.OUTPUT_CHANNELS_PER_BLOCK,
                                                        blocks_count=configs.B_STACK_COUNT,
                                                        )

regressionHead= model_setup.getRegressionHead(input_shape=configs.REGRESSION_INPUT_SHAPE,
                                                output_size=configs.REGRESSION_OUTPUT_SHAPE,
                                                blocks_count=configs.R_STACK_COUNT,
                                                )


today = date.today()
Utils.plot_and_save_model_structure(featureEmbeddingBackBone,
                                            save_path="resources/",
                                            save_as=f"featureEmbeddingBackBone-{today}")
Utils.plot_and_save_model_structure(regressionHead,
                                            save_path="resources/",
                                            save_as=f"regressionHead-{today}")

