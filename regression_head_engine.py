import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

# from tqdm.auto import tqdm    

import Tools.backboneUtils as backboneUtils
import Tools.loss_functions as loss_functions
import Tools.datasetTools as DatasetTools
import Tools.utilities as common_utils
from collections import defaultdict




# train step for regression head
def train_step(model,
                backBone,
                dataloader,
                optimizer,
                predicting_homography):  
    """
    A train step for the regression head
    """
    
    dataloader = dataloader.shuffle(1000)   
        
    model.compile(optimizer=optimizer) 
    epochs_losses_summary= {"backbone": defaultdict(list),
                            "regression_head": defaultdict(list)
                            }
    
    assert backBone is not None, "the feature embedding backbone is not defined"
    
    for i, batch in enumerate(dataloader):
        input_images, template_images, labels,_instances = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        if len(input_images.shape) != 4:
            input_images = tf.expand_dims(input_images, axis=0)
            template_images  = tf.expand_dims(template_images, axis=0)
            labels = tf.expand_dims(labels, axis=0)
        
        # pass the input images through the backbone
        
        
        gt_matrix=DatasetTools.get_ground_truth_homographies(labels)
        warped_inputs, _ = DatasetTools._get_warped_sampled(input_images, gt_matrix)
        
        rgb_fmaps , ir_fmaps = backBone.call((input_images, template_images),training=False)
        
        # padd the ir_fmaps to match the shape of the rgb_fmaps
        
        
        ir_fmaps_padded = backboneUtils.get_padded_fmaps(fmaps=ir_fmaps, desired_shape = rgb_fmaps.shape)
        # concatenate the rgb_fmaps and ir_fmaps
        concatenated_fmaps = tf.concat([rgb_fmaps, ir_fmaps_padded], axis=-1)
        
        with tf.GradientTape() as tape:
                predictions = model.call((concatenated_fmaps),training=True)
                
                total_loss , detailed_batch_losses = loss_functions.get_losses_regression_head( predictions = predictions, 
                                                                                                ground_truth_corners = labels,
                                                                                                gt_matrix = gt_matrix, 
                                                                                                predicting_homography=predicting_homography)
                assert not np.isnan(total_loss.numpy()), "Loss is NaN"
                
        all_parameters= model.trainable_variables
        grads = tape.gradient(total_loss, all_parameters)
        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))
        
        # add losses to epoch losses
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary["regression_head"][key].append(value)
            
            
        # backbone losses 
        warped_fmaps,_ = DatasetTools._get_warped_sampled(rgb_fmaps, gt_matrix)                        
        total_loss_backbone , detailed_batch_losses_backbone = loss_functions.get_losses_febackbone(warped_inputs,
                                                                                                    template_images,
                                                                                                    warped_fmaps,
                                                                                                    ir_fmaps)
        # loss shouldn't be nan
        assert not np.isnan(total_loss_backbone.numpy()), "Loss is NaN"    
        # add losses to epoch losses
        for key, value in detailed_batch_losses_backbone.items():
            epochs_losses_summary["backbone"][key].append(value)  
        # log = " ".join([str(i + " :" + k + "\n") for i,k in detailed_batch_losses.items()])
        # print(log)
    
    # compute mean of losses
    for step, losses in epochs_losses_summary.items():
        for key, value in losses.items():
            epochs_losses_summary[step][key] = np.mean(value)

    # display losses
    log = " | ".join([str(str(i)+ " : " + str(k)) for i,k in epochs_losses_summary["regression_head"].items()])
    print(f"[train_loss] : {log}")
    return epochs_losses_summary


# test step for the regression head
def test_step(model,
                backBone,
                dataloader,
                predicting_homography): 
    """
    Test step for the regression head
    """
    epochs_losses_summary= {"backbone": defaultdict(list),
                            "regression_head": defaultdict(list)
                            }
    
    assert backBone is not None, "the feature embedding backbone is not defined"
    
    for i, batch in enumerate(dataloader):
        input_images, template_images, labels,_instances = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        if len(input_images.shape) != 4:
            input_images = tf.expand_dims(input_images, axis=0)
            template_images  = tf.expand_dims(template_images, axis=0)
            labels = tf.expand_dims(labels, axis=0)
        
        # pass the input images through the backbone
        
        
        gt_matrix=DatasetTools.get_ground_truth_homographies(labels)
        warped_inputs, _ = DatasetTools._get_warped_sampled(input_images, gt_matrix)
        
        rgb_fmaps , ir_fmaps = backBone.call((input_images, template_images),training=False)
        
        # padd the ir_fmaps to match the shape of the rgb_fmaps
        
        
        ir_fmaps_padded = backboneUtils.get_padded_fmaps(fmaps=ir_fmaps, desired_shape = rgb_fmaps.shape)
        # concatenate the rgb_fmaps and ir_fmaps
        concatenated_fmaps = tf.concat([rgb_fmaps, ir_fmaps_padded], axis=-1)
        

        predictions = model.call((concatenated_fmaps),training=False)
        total_loss , detailed_batch_losses = loss_functions.get_losses_regression_head( predictions = predictions, 
                                                                                        ground_truth_corners = labels,
                                                                                        gt_matrix = gt_matrix, 
                                                                                        predicting_homography=predicting_homography)
        assert not np.isnan(total_loss.numpy()), "Loss is NaN"
        # add losses to epoch losses
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary["regression_head"][key].append(value)
            
        # backbone losses 
        warped_fmaps,_ = DatasetTools._get_warped_sampled(rgb_fmaps, gt_matrix)                        
        total_loss_backbone , detailed_batch_losses_backbone = loss_functions.get_losses_febackbone(warped_inputs,
                                                                                                    template_images,
                                                                                                    warped_fmaps,
                                                                                                    ir_fmaps)
        # loss shouldn't be nan
        assert not np.isnan(total_loss_backbone.numpy()), "Loss is NaN"    
        # add losses to epoch losses
        for key, value in detailed_batch_losses_backbone.items():
            epochs_losses_summary["backbone"][key].append(value)
    
    # compute mean of losses
    for step, losses in epochs_losses_summary.items():
        for key, value in losses.items():
            epochs_losses_summary[step][key] = np.mean(value)

    # display losses
    log = " | ".join([str(str(i)+ " : " + str(k)) for i,k in epochs_losses_summary["regression_head"].items()])
    print(f"[test_loss] : {log}")
    return epochs_losses_summary