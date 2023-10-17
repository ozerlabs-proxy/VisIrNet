## VisIrNet

## setup

```bash 

    #env
    conda create -n VisIrNet python==3.10
    conda activate VisIrNet
    pip install --upgrade pip 
    conda install -c anaconda cudatoolkit
    conda install -c anaconda cudnn
    pip install tensorflow[and-cuda]

    #packages
    conda install pillow

    # model visualization
    conda install graphviz
    conda install python-graphviz
    conda install pydot

    conda install -c conda-forge tqdm
    #  pip install tf-geometric
    pip install tensorflow_graphics

    pip install opencv-contrib-python

```

### Data


```bash
# create data under VisIrNet/
mkdir data
cd data

##1. you can run the linking script
# python ./scripts/link_datasets_to_data.py
#2 create symbolic links to datasets

cd data 
ln -s ~/ozerlabs-workspace/Datasets/registration-datasets/GoogleEarth .
ln -s ~/ozerlabs-workspace/Datasets/registration-datasets/MSCOCO .
ln -s ~/ozerlabs-workspace/Datasets/registration-datasets/SkyData .
ln -s ~/ozerlabs-workspace/Datasets/registration-datasets/VEDAI .
ln -s ~/ozerlabs-workspace/Datasets/registration-datasets/GoogleMap .
```


## Training

change configuration file in `training.sh`

```bash

...
# This script is used to train the model.

conda activate VisIrNet
python Train.py --config-file >>>skydata_default_config.json<<<
# python Train.py --config-file >>>vedai_default_config.json
...

```

```bash
## if running local
bash training.sh

## if running on slurm
sbatch trainingJob.sh
```

## Visualize plots

visualize logs from tensorboard

```bash
# make sure conda env is activated
conda activate VisIrNet
tensorboard --logdir logs/tensorboard
```