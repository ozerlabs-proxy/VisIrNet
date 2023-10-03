# ##
import os
import sys
import tensorflow as tf
from pathlib import Path
import numpy as np
import PIL
import PIL.Image
import json

import warnings
warnings.filterwarnings('ignore')

# print(tf.__version__)
# devices = tf.config.list_physical_devices('GPU')
# print("len(devices): ", len(devices))
# print(f"available GPUs: {devices}");


# ##
#change working directory to root
ROOT_DIR = os.getcwd()
while os.path.basename(ROOT_DIR) != 'VisIrNet':
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR,'..'))
sys.path.insert(0,ROOT_DIR)
os.chdir(ROOT_DIR)

ROOT_DIR = Path(ROOT_DIR)
DATA_DIR = ROOT_DIR / 'data'


# ##
""" DataLoaders """
from Tools.data_read import Dataset
# try to import the dataset
dataset="SkyData"
BATCHSIZE = 2
SHUFFLE_BUFFER_SIZE = 100

train_dataset = Dataset(dataset=dataset,split="train").dataset
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCHSIZE,drop_remainder=True)

val_dataset = Dataset(dataset=dataset,split="val").dataset
val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCHSIZE,drop_remainder=True)


# ##


