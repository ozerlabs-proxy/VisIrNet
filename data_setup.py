"""
Contains functionality for creating the dataset and dataloader for skydata.

the dataset class is SkyDataDET and it is a subclass of torch.utils.data.Dataset class. It takes much after the COCO dataset class from torchvision. It is used to load the annotations and images from the skydata dataset and prepare them for training and evaluation. 

we will have train_loader, test_loader 
"""

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import json
from pathlib import Path

from Tools._Datasets import *

def create_dataloaders(
                    dataset: str="SkyData", 
                    BATCH_SIZE = None,
                    SHUFFLE_BUFFER_SIZE=None
                    ):
    """_summary_

    Args:
        dataset (str, optional): _description_. Defaults to "SkyData".
        BATCH_SIZE (int, optional): _description_. Defaults to 1.
        SHUFFLE_BUFFER_SIZE (int, optional): _description_. Defaults to 100.

    Returns:
        train_dataloader, test_dataloader
    """
    

    assert dataset in ["SkyData", "VEDAI","GoogleEarth","GoogleMap","MSCOCO"] , "dataset not supported"
    assert BATCH_SIZE is not None, "BATCH_SIZE is not defined"
    assert SHUFFLE_BUFFER_SIZE is not None, "SHUFFLE_BUFFER_SIZE is not defined"
    
    
    train_dataset = Dataset(dataset = dataset,
                                split="train").dataset
    test_dataset = Dataset(dataset = dataset,
                            split="val").dataset
    
    train_dataloader= train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    test_dataloader = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    print(f"dataset: {dataset}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"SHUFFLE_BUFFER_SIZE: {SHUFFLE_BUFFER_SIZE}")
    print(f"train_dataloader: {len(train_dataloader)}")
    print(f"test_dataloader: {len(test_dataloader)}")

    return train_dataloader, test_dataloader


def create_dataloader_test(
                        dataset: str="SkyData", 
                        BATCH_SIZE = 1,
                        ):
    """
    _summary_
        Generate and return a dataloader for the test dataset given dataset name and batch size

    Returns:
        train_dataloader, test_dataloader
    """
    

    assert dataset in ["SkyData", "VEDAI","GoogleEarth","GoogleMap","MSCOCO"] , "dataset not supported"
    assert BATCH_SIZE is not None, "BATCH_SIZE is not defined"
    
    test_dataset = Dataset(dataset = dataset, split="val").dataset
    
    test_dataloader = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
    
    print(f"dataset: {dataset}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"test_dataloader: {len(test_dataloader)}")

    return test_dataloader