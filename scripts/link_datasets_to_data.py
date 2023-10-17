"""
Script will link datasets to the data folder, which we will work with.
"""


""" you can alternatively run the following commands in the terminal
    cd data 
    ln -s ~/ozerlabs-workspace/Datasets/GoogleEarth .
    ln -s ~/ozerlabs-workspace/Datasets/MSCOCO .
    ln -s ~/ozerlabs-workspace/Datasets/SkyData .
    ln -s ~/ozerlabs-workspace/Datasets/VEDAI .
    ln -s ~/ozerlabs-workspace/Datasets/GoogleMap .

"""

import os
import sys
from pathlib import Path


# import warnings
# warnings.filterwarnings('ignore')


# print(tf.__version__)
# devices = tf.config.list_physical_devices('GPU')
# print("len(devices): ", len(devices))
# print(f"available GPUs: {devices}")

# ##
#change working directory to root
ROOT_DIR = os.getcwd()
while os.path.basename(ROOT_DIR) != 'VisIrNet':
    ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR,'..'))
sys.path.insert(0,ROOT_DIR)
os.chdir(ROOT_DIR)

# ##
# lets create symbolic links to the data

assert Path.cwd().name == "VisIrNet", "You need to be in the root directory of the project"
SOURCE_PATH = Path.home() / "ozerlabs-workspace/Datasets/"
datasets = list(SOURCE_PATH.glob("*"))
DATA_DIR = Path.cwd() / "data"


# ##
## attemp to create symbolic links
for dataset in datasets:
    if dataset.is_dir():
        os.symlink(dataset, DATA_DIR / dataset.name)

