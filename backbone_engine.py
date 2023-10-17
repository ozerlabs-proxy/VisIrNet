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

# tf.keras.mixed_precision.set_global_policy('mixed_float16')



# train step for the feature embedding backbone
def train_step(model,
                    dataloader,
                    optimizer):  
    
    dataloader = dataloader.shuffle(1000)   
    # model.compile(optimizer=optimizer , experimental_run_tf_function=False) 
    model.compile(optimizer=optimizer) 
    epochs_losses_summary= defaultdict(list)
    
    print(f"[INFO] training  on {len(dataloader)} pairs")
    for i, batch in tqdm(enumerate(dataloader.take(64))):
        input_images, template_images, labels,_instances = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        if len(input_images.shape) != 4:
            
            print(f'[unmatching shape] : {input_images.shape}')
            print(f'[unmatching shape] : {template_images.shape}')
            print(f'[unmatching shape] : {labels.shape}')
            
            input_images = tf.expand_dims(input_images, axis=0) if len(input_images.shape) == 3 else input_images
            template_images  = tf.expand_dims(template_images, axis=0) if len(template_images.shape) == 3 else template_images
            labels = tf.expand_dims(labels, axis=0) if len(labels.shape) == 1 else labels
            

        
        gt_matrix = DatasetTools.get_ground_truth_homographies(labels)

        
        assert len(input_images.shape) == 4, "input_images shape is not (batch_size, height, width, channels)"
        assert len(template_images.shape) == 4, "template_images shape is not (batch_size, height, width, channels)"
        assert len(labels.shape) == 2, "labels shape is not (batch_size, uv_list/homography)"
        assert gt_matrix.shape == (input_images.shape[0], 3, 3), "gt_matrix shape is not (batch_size, 3, 3)"
        
        warped_inputs, _ = DatasetTools._get_warped_sampled(images = input_images,  homography_matrices = gt_matrix)
        assert len(warped_inputs.shape) == len(input_images.shape), "warped_inputs shape is not (batch_size, height, width, channels)"
        
        with tf.GradientTape() as tape:
                rgb_fmaps , ir_fmaps = model.call((input_images, template_images), training=True)
                
                
                # assert rgb_fmaps.shape == (input_images.shape[0], 192, 192, 3), f"rgb_fmaps shape is not {rgb_fmaps.shape}"
                # assert ir_fmaps.shape == (input_images.shape[0], 128, 128, 3), f"ir_fmaps shape is not {ir_fmaps.shape}"
                
                warped_fmaps,_ = DatasetTools._get_warped_sampled( images = rgb_fmaps, 
                                                                    homography_matrices = gt_matrix)
                                
                total_loss , detailed_batch_losses = loss_functions.get_losses_febackbone(warped_inputs,
                                                                                            template_images,
                                                                                            warped_fmaps,
                                                                                            ir_fmaps)
                
                                                            
                
        all_parameters= model.trainable_variables
        # assert tf.math.is_finite(all_parameters).all(), "all_parameters in backbone are inf or NaN"
        grads = tape.gradient(total_loss, all_parameters,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads]        
        # assert tf.math.is_finite(grads).all(), "Gradients in backbone are inf or NaN"
        optimizer.apply_gradients(zip(grads, all_parameters))
        
        # add losses to epoch losses
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary[key].append(value)
            
        # log = " ".join([str(i + " :" + k + "\n") for i,k in detailed_batch_losses.items()])
        # print(log)
        
        ##  reset keras cache
        tf.keras.backend.clear_session()
        input_images, template_images, labels,_instances = None, None, None, None
        warped_inputs, rgb_fmaps, ir_fmaps, warped_fmaps = None, None, None, None
        # total_loss, detailed_batch_losses = None, None
        
    # compute mean of losses
    for key, value in epochs_losses_summary.items():
        epochs_losses_summary[key] = np.mean(value)
    
    # display losses
    # display losses
    log = " | ".join([str(str(i)+ " : " + str(k)) for i,k in epochs_losses_summary.items()])
    print(f"[train_loss] : {log}")
    return model, epochs_losses_summary



# test step for the feature embedding backbone
def test_step(model,
                dataloader): 
    
    epochs_losses_summary= defaultdict(list)
    print(f"[INFO] testing  on {len(dataloader)} pairs")
    for i, batch in enumerate(dataloader.take(64)):
        input_images, template_images, labels,_instances = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        if len(input_images.shape) != 4:            
            print(f'[unmatching shape] : {input_images.shape}')
            print(f'[unmatching shape] : {template_images.shape}')
            print(f'[unmatching shape] : {labels.shape}')
            
            input_images = tf.expand_dims(input_images, axis=0)
            template_images  = tf.expand_dims(template_images, axis=0)
            labels = tf.expand_dims(labels, axis=0)
            
        assert len(input_images.shape) == 4, "input_images shape is not (batch_size, height, width, channels)"
        assert len(template_images.shape) == 4, "template_images shape is not (batch_size, height, width, channels)"
        assert len(labels.shape) == 2, "labels shape is not (batch_size, uv_list/homography)"
        
        
        gt_matrix=DatasetTools.get_ground_truth_homographies(labels)
        assert gt_matrix.shape == (input_images.shape[0], 3, 3), "gt_matrix shape is not (batch_size, 3, 3)"
        
        warped_inputs, _ = DatasetTools._get_warped_sampled(images = input_images, 
                                                            homography_matrices = gt_matrix)
        
        rgb_fmaps , ir_fmaps = model.call((input_images, template_images), training=False)
        warped_fmaps, _  = DatasetTools._get_warped_sampled(images = rgb_fmaps, 
                                                            homography_matrices = gt_matrix)
        
        total_loss , detailed_batch_losses = loss_functions.get_losses_febackbone(warped_inputs,
                                                                                    template_images,
                                                                                    warped_fmaps,
                                                                                    ir_fmaps)
            
        # loss shouldn't be nan
        assert np.isfinite(total_loss).all(), "Loss is NaN"
        
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
