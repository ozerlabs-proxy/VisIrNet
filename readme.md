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

## Notebooks

- [Train Test Flow](notebooks/pipeline.ipynb)
- [Playground](notebooks/playground.ipynb)
- [Visualize fmaps of backbone](notebooks/visualizeBackBoneRes.ipynb)
- [Visualize fmaps of backbone](notebooks/visualizeBackBoneRes.ipynb)


## loss functions logs
### backbone losses choice
- &check; mean_squared_error (mse_pixel) "l2"
- &check; mean_absolute_error (mae_pixel) "l1"
- &check; sum_squared_error (sse_pixel)
- &check; structural_similarity (ssim_pixel)


### registration losses choice

which ever used you have access to other derrivations

- [x] l1_homography_loss
- [x] l2_homography_loss
- [x] l1_corners_loss
- [x] l2_corners_loss

## TRACK EXPERIMENTS

### BACKBONEs 
a table of backbones with diferent losses and datasets

|           |SkyData    |VEDAI  |
|-----------|-----------|-------|
| mse_pixel | &check;       | &check;   | 
| mae_pixel | &check;       | &check;   |  
| sse_pixel | &check;       | &check;   |
| ssim_pixel| &check;       | &check;   | 


### regression head
for each dataset there will be regression head trained on different backbones with different regression losses

| Backbone  | R_loss | SkyData    |VEDAI  |
|-----------|-----------------|-----------|-------|
| mse_pixel | l2_corners_loss | &check;       | &check;   | 
| mae_pixel | l2_corners_loss | &check;       | &check;   | 
| sse_pixel | l2_corners_loss | &check;       | &check;   | 
| ssim_pixel| l2_corners_loss | &check;       | &check;   |
***

| Backbone  | R_loss | SkyData    |VEDAI  |
|-----------|-----------------|-----------|-------|
| mse_pixel | l2_homography_loss | -      | -   | 
| mae_pixel | l2_homography_loss | -      | -   |
| sse_pixel | l2_homography_loss | -      | -   |
| ssim_pixel| l2_homography_loss | -      | -   | 


<!-- 
| Backbone  | R_loss | SkyData    |VEDAI  |GoogleEarth    |GoogleMap  |MSCOCO |
|-----------|-----------------|-----------|-------|---------------|-----------|-------|
| mse_pixel | l1_homography_loss | &check;       | -   | -           | -       | -   |
| mae_pixel | l1_homography_loss | &check;       | -   | -           | -       | -   |
| sse_pixel | l1_homography_loss | &check;       | -   | -           | -       | -   |
| ssim_pixel| l1_homography_loss | &check;       | -   | -           | -       | -   | -->
***
<!-- | Backbone  | R_loss | SkyData    |VEDAI  |GoogleEarth    |GoogleMap  |MSCOCO |
|-----------|--------|------------|-------|---------------|-----------|-------|
| mse_pixel | l1_corners_loss | &check;       | -   | -           | -       | -   |
| mae_pixel | l1_corners_loss | &check;       | -   | -           | -       | -   |
| sse_pixel | l1_corners_loss | &check;       | -   | -           | -       | -   |
| ssim_pixel| l1_corners_loss | &check;       | -   | -           | -       | -   | 
*** -->



```bash 
/cta/users/ndigande/ozerlabs-workspace/VisIrNet
sbatch slurm-vedai-Job-test-purposes.sh
```