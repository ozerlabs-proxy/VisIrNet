#  ##
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


#  ## [markdown]
# **Dataloaders**

#  ##
import data_setup

# try to import the dataset
dataset="SkyData"
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 100

train_dataloader,test_dataloader = data_setup.create_dataloaders(dataset=dataset, 
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                SHUFFLE_BUFFER_SIZE=100
                                                                )
#

#  ## [markdown]
# ## **Model**

#  ##
import model_setup
import Utils
rgb_inputs_shape = (192,192,3)
ir_inputs_shape =  (128,128,3)
output_channels_per_block = 3
regression_input_shape = (*rgb_inputs_shape[:2],output_channels_per_block*2)
regression_output_shape = 8


#  ##
featureEmbeddingBackBone = model_setup.getFeatureEmbeddingBackBone(rgb_inputs_shape=rgb_inputs_shape,
                                                        ir_inputs_shape=ir_inputs_shape,
                                                        output_channels_per_block=output_channels_per_block
                                                        )

regressionHead= model_setup.getRegressionHead(input_shape=regression_input_shape,
                                                output_size=regression_output_shape
                                                )


#  ## [markdown]
# **Visualize and save model structures**

#  ##
# # visualize and save models

# Utils.plot_and_save_model_structure(featureEmbeddingBackBone,
#                                             save_path="resources/",
#                                             save_as=f"featureEmbeddingBackBone")
# Utils.plot_and_save_model_structure(regressionHead,
#                                             save_path="resources/",
#                                             save_as=f"regressionHead")

#  ## [markdown]
# ## **Training**
# 

#  ##
import engine 

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                decay_steps=10000,
                                                                decay_rate=0.96,
                                                                staircase=True)
NUM_EPOCHS = 10

# Setup optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Start the timer
from timeit import default_timer as timer
start_time = timer()
# Train model 

model_results = engine.train_first_stage(model=featureEmbeddingBackBone,
                                                train_dataloader=train_dataloader,
                                                test_dataloader=test_dataloader,
                                                optimizer=optimizer,
                                                epochs=NUM_EPOCHS,
                                                save_path="models",
                                                save_as=f"featureEmbeddingBackBone",
                                                save_frequency=1,
                                                save_hard_frequency=50
                                                )
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time : {end_time-start_time:.3f} seconds")

