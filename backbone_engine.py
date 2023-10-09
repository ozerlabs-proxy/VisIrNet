import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

from tqdm.auto import tqdm    

import Tools.backboneUtils as backboneUtils
import Tools.loss_functions as loss_functions
import Tools.datasetTools as DatasetTools
import Tools.utilities as common_utils
from collections import defaultdict




# train step for the feature embedding backbone
def train_step(model,
                            dataloader,
                            optimizer):  
    
    dataloader = dataloader.shuffle(1000)   
    model.compile(optimizer=optimizer) 
    epochs_losses_summary= defaultdict(list)
    
    for i, batch in tqdm(enumerate(dataloader.take(8))):
        input_images, template_images, labels,_instances = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        if len(input_images.shape) != 4:
            input_images = tf.expand_dims(input_images, axis=0)
            template_images  = tf.expand_dims(template_images, axis=0)
            labels = tf.expand_dims(labels, axis=0)
        
        gt_matrix=DatasetTools.get_ground_truth_homographies(labels)
        warped_inputs, _ = DatasetTools._get_warped_sampled(input_images, gt_matrix)
        
        with tf.GradientTape() as tape:
                rgb_fmaps , ir_fmaps=model.call((input_images, template_images),training=True)
                warped_fmaps,_ = DatasetTools._get_warped_sampled(rgb_fmaps, gt_matrix)
                                
                total_loss , detailed_batch_losses = loss_functions.get_losses_febackbone(warped_inputs,template_images,warped_fmaps,ir_fmaps)
                # loss shouldn't be nan
                assert not np.isnan(total_loss.numpy()), "Loss is NaN"
                
        all_parameters= model.trainable_variables
        grads = tape.gradient(total_loss, all_parameters)
        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))
        
        # add losses to epoch losses
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary[key].append(value)
            
        # log = " ".join([str(i + " :" + k + "\n") for i,k in detailed_batch_losses.items()])
        # print(log)
    
    # compute mean of losses
    for key, value in epochs_losses_summary.items():
        epochs_losses_summary[key] = np.mean(value)
    
    # display losses
    # display losses
    log = " | ".join([str(str(i)+ " : " + str(k)) for i,k in epochs_losses_summary.items()])
    print(f"[train_loss] : {log}")
    return epochs_losses_summary
# test step for the feature embedding backbone
def test_step(model,
                        dataloader): 
    epochs_losses_summary= defaultdict(list)
    for i, batch in enumerate(dataloader.take(8)):
        input_images, template_images, labels,_instances = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        if len(input_images.shape) != 4:
            input_images = tf.expand_dims(input_images, axis=0)
            template_images  = tf.expand_dims(template_images, axis=0)
            labels = tf.expand_dims(labels, axis=0)
        
        gt_matrix=DatasetTools.get_ground_truth_homographies(labels)
        warped_inputs, _ = DatasetTools._get_warped_sampled(input_images, gt_matrix)
        
        rgb_fmaps , ir_fmaps=model.call((input_images, template_images), training=False)
        warped_fmaps,_ = DatasetTools._get_warped_sampled(rgb_fmaps, gt_matrix)
        
        total_loss , detailed_batch_losses = loss_functions.get_losses_febackbone(warped_inputs,template_images,warped_fmaps,ir_fmaps)
        
        # loss shouldn't be nan
        assert not np.isnan(total_loss.numpy()), "Loss is NaN"
        
        # add losses to epoch losses
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary[key].append(value)
            
        # log = " ".join([str(i + " :" + k + "\n") for i,k in batch_losses.items()])
        # print(log)
    
    # compute mean of losses
    for key, value in epochs_losses_summary.items():
        epochs_losses_summary[key] = np.mean(value)
    
    # display losses
    log = " | ".join([str(str(i)+ " :" + str(k)) for i,k in epochs_losses_summary.items()])
    print(f"[test_loss] : {log}")
    
    return epochs_losses_summary
