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


# ## [markdown]
# **Dataloaders**

# ##
import data_setup

# try to import the dataset
dataset="SkyData"
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 100

train_dataloader,test_dataloader = data_setup.create_dataloaders(dataset=dataset, 
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                SHUFFLE_BUFFER_SIZE=100
                                                                )

len(train_dataloader), len(test_dataloader)
#

# ## [markdown]
# ## **Model**

# ##
import model_setup
import Utils

# ## [markdown]
# **configuration**
# 
# 
# or load configuration from a file

# ##
# config constants 
configs = {
            'RGB_INPUTS_SHAPE' : (192,192,3),
            'IR_INPUTS_SHAPE' :  (128,128,3),
            'OUTPUT_CHANNELS_PER_BLOCK' : 3,
            'REGRESSION_INPUT_SHAPE' : None,
            'REGRESSION_OUTPUT_SHAPE' : 8    
            }
configs['REGRESSION_INPUT_SHAPE']= (*configs["RGB_INPUTS_SHAPE"][:2], configs["OUTPUT_CHANNELS_PER_BLOCK"]*2)
assert configs['REGRESSION_INPUT_SHAPE'] != None

# ##
featureEmbeddingBackBone = model_setup.getFeatureEmbeddingBackBone(rgb_inputs_shape=configs['RGB_INPUTS_SHAPE'],
                                                        ir_inputs_shape=configs['IR_INPUTS_SHAPE'],
                                                        output_channels_per_block=configs['OUTPUT_CHANNELS_PER_BLOCK']
                                                        )

regressionHead= model_setup.getRegressionHead(input_shape=configs['REGRESSION_INPUT_SHAPE'],
                                                output_size=configs['REGRESSION_OUTPUT_SHAPE']
                                                )


# ## [markdown]
# **Visualize and save model structures**

# ##
# # visualize and save models

# Utils.plot_and_save_model_structure(featureEmbeddingBackBone,
#                                             save_path="resources/",
#                                             save_as=f"featureEmbeddingBackBone")
# Utils.plot_and_save_model_structure(regressionHead,
#                                             save_path="resources/",
#                                             save_as=f"regressionHead")

# ## [markdown]
# ## **Training**
# 

# ## [markdown]
# **first stage**

# ##
import engine 

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                decay_steps=10000,
                                                                decay_rate=0.96,
                                                                staircase=True)
NUM_EPOCHS = 5

# Setup optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Start the timer
from timeit import default_timer as timer
start_time = timer()
# Train model 

model_results = engine.train_first_stage(model=featureEmbeddingBackBone,
                                                train_dataloader=train_dataloader,
                                                test_dataloader=test_dataloader,
                                                dataset_name=dataset,
                                                optimizer=optimizer,
                                                epochs=NUM_EPOCHS,
                                                from_checkpoint=None,
                                                save_path=f"models/{dataset}",
                                                save_as=f"featureEmbeddingBackBone",
                                                save_frequency=1,
                                                save_hard_frequency=50
                                                )
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time : {end_time-start_time:.3f} seconds")

# ## [markdown]
# # **second stage**

##
import engine 

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                decay_steps=10000,
                                                                decay_rate=0.96,
                                                                staircase=True)
NUM_EPOCHS = 2

# Setup optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Start the timer
from timeit import default_timer as timer
start_time = timer()
# Train model 

model_results = engine.train_second_stage(model=regressionHead,
                                        featureEmbeddingBackBone="latest",
                                        train_dataloader=train_dataloader,
                                        test_dataloader=test_dataloader,
                                        dataset_name=dataset,
                                        optimizer=optimizer,
                                        epochs=NUM_EPOCHS,
                                        from_checkpoint=None,
                                        save_path=f"models/{dataset}",
                                        save_as=f"regressionHead",
                                        save_frequency=1,
                                        save_hard_frequency=20,
                                        predicting_homography=True
                                        )
# End the timer and print out how long it took
end_time = timer()
print(f"Total training time : {end_time-start_time:.3f} seconds")

