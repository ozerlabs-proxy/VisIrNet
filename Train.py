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
parser.add_argument('--config-file', 
                        action = "store", 
                        dest = "config_file",
                        default = "default_config.json",
                        help = 'specify config file to load')

input_arguments = parser.parse_args()

from Tools.configurations_parser import ConfigurationParser
# load configurations
configs = ConfigurationParser.getConfigurations(configs_path = 'configs', 
                                                config_file = str(input_arguments.config_file))


# print configurations
ConfigurationParser.printConfigurations(configs)


# ## [markdown]
# **Dataloaders**

# ##
import data_setup

train_dataloader, test_dataloader = data_setup.create_dataloaders(dataset = configs.dataset, 
                                                                  BATCH_SIZE = configs.BATCH_SIZE,
                                                                  SHUFFLE_BUFFER_SIZE = configs.SHUFFLE_BUFFER_SIZE
                                                                  )

print(f"{len(train_dataloader)} , {len(test_dataloader)}")
#

# ## [markdown]
# ## **Model**

# ##
import model_setup
import Utils

featureEmbeddingBackBone = model_setup.getFeatureEmbeddingBackBone(rgb_inputs_shape=configs.RGB_INPUTS_SHAPE,
                                                        ir_inputs_shape=configs.IR_INPUTS_SHAPE,
                                                        output_channels_per_block=configs.OUTPUT_CHANNELS_PER_BLOCK,
                                                        blocks_count=configs.B_STACK_COUNT,
                                                        )

regressionHead= model_setup.getRegressionHead(input_shape=configs.REGRESSION_INPUT_SHAPE,
                                                output_size=configs.REGRESSION_OUTPUT_SHAPE,
                                                blocks_count=configs.R_STACK_COUNT,
                                                )


# ## [markdown]
# **Visualize and save model structures**
# 

# ## [markdown]
# 
# ```python
# 
# # visualize and save models
# 
# Utils.plot_and_save_model_structure(featureEmbeddingBackBone,
#                                             save_path="resources/",
#                                             save_as=f"featureEmbeddingBackBone")
# Utils.plot_and_save_model_structure(regressionHead,
#                                             save_path="resources/",
#                                             save_as=f"regressionHead")
# 
# ```

# ## [markdown]
# ## **Training**
# 

# ## [markdown]
# **first stage**

# ##
import engine 

if configs.TrainFirstStage:
    print("*"*25, f"first stage", "*"*25)
    print(f"uuid: {configs.B_R_uuid}")
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=configs.B_initial_learning_rate,
                                                                    decay_steps=configs.B_decay_steps,
                                                                    decay_rate=configs.B_decay_rate,
                                                                    staircase=True)


    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()
    # Train model 

    featureEmbeddingBackBone, model_results = engine.train_first_stage(model=featureEmbeddingBackBone,
                                                            train_dataloader=train_dataloader,
                                                            test_dataloader=test_dataloader,
                                                            dataset_name=configs.dataset,
                                                            optimizer=optimizer,
                                                            epochs=configs.B_NUM_EPOCHS,
                                                            from_checkpoint=configs.B_from_checkpoint,
                                                            save_path=configs.B_save_path,
                                                            save_as=configs.B_save_as,
                                                            save_frequency=configs.B_save_frequency,
                                                            save_hard_frequency=configs.B_save_hard_frequency,
                                                            uuid=configs.B_R_uuid
                                                            )
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time : {end_time-start_time:.3f} seconds\n\n")

# ## [markdown]
# **second stage**

# ##
import engine 

if configs.TrainSecondStage:
    print("*"*25, f"second stage", "*"*25)
    print(f"uuid: {configs.B_R_uuid}")
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=configs.R_initial_learning_rate,
                                                                    decay_steps=configs.R_decay_steps,
                                                                    decay_rate=configs.R_decay_rate,
                                                                    staircase=True)


    # Setup optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()
    # Train model 

    regressionHead , model_results = engine.train_second_stage(model = regressionHead,
                                                    featureEmbeddingBackBone = configs.R_featureEmbeddingBackBone,
                                                    train_dataloader = train_dataloader,
                                                    test_dataloader = test_dataloader,
                                                    dataset_name = configs.dataset,
                                                    optimizer = optimizer,
                                                    epochs = configs.R_NUM_EPOCHS,
                                                    from_checkpoint = configs.R_from_checkpoint,
                                                    save_path = configs.R_save_path,
                                                    save_as = configs.R_save_as,
                                                    save_frequency = configs.R_save_frequency,
                                                    save_hard_frequency = configs.R_save_hard_frequency,
                                                    predicting_homography = configs.R_predicting_homography,
                                                    uuid = configs.B_R_uuid
                                                    )
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time : {end_time-start_time:.3f} seconds\n\n")

# ##


# ##


# ##



