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
tf.keras.mixed_precision.set_global_policy('mixed_float16')



# train step for regression head
def train_step(model,
                backBone,
                dataloader,
                optimizer,
                predicting_homography,
                backbone_loss_function,
                loss_function_to_use):  
    """
    A train step for the regression head
    """
    
    dataloader = dataloader.shuffle(1000)   
        
    # model.compile(optimizer=optimizer,experimental_run_tf_function=False) 
    epochs_losses_summary= {"backbone": defaultdict(list),
                            "regression_head": defaultdict(list)
                            }
    
    assert backBone is not None, "the feature embedding backbone is not defined"
    print(f"[INFO] training  on {len(dataloader)} pairs")
    for i, batch in tqdm(enumerate(dataloader)):
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
        

        # weighted sum if needed 
        summed_rgb_fmaps = tf.constant(1.0) * tf.cast(input_images,"float")   + tf.constant(0.2) * tf.cast(rgb_fmaps,"float")
        summed_ir_fmaps =  tf.constant(0.2) * tf.cast(template_images,"float") + tf.constant(1.0) *  tf.cast(ir_fmaps,'float')
        # padd the ir_fmaps to match the shape of the rgb_fmaps
        ir_fmaps_padded = backboneUtils.get_padded_fmaps(fmaps=summed_ir_fmaps, desired_shape = rgb_fmaps.shape)

        
        # concatenate the rgb_fmaps and ir_fmaps
        concatenated_fmaps = tf.concat([summed_rgb_fmaps, ir_fmaps_padded], axis=-1)
        
        with tf.GradientTape() as tape:
                predictions = model.call((concatenated_fmaps), training=True)
                
                tape.watch(predictions)
                total_loss , detailed_batch_losses = loss_functions.get_losses_regression_head( predictions = predictions, 
                                                                                                ground_truth_corners = labels,
                                                                                                gt_matrix = gt_matrix, 
                                                                                                predicting_homography=predicting_homography,
                                                                                                loss_function_to_use = loss_function_to_use)
                # assert tf.reduce_all(tf.math.is_finite(total_loss)), "Loss is NaN"
                
        all_parameters= model.trainable_variables
        try:
            grads = tape.gradient(total_loss, all_parameters, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        except Exception as e:
            # somestupid error
            print("*"*100)
            print(f"[ERROR] ----------------------- skipping batch {i} --------------------------")
            print("*"*100)
            continue
        
        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads]
        # assert tf.reduce_all(tf.math.is_finite(grads)), "Gradients in regression head are inf or NaN"
        optimizer.apply_gradients(zip(grads, all_parameters))
        
        # add losses to epoch losses
        detailed_batch_losses = {key: value.numpy() for key, value in detailed_batch_losses.items()}
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary["regression_head"][key].append(value)
            
            
        # backbone losses 
        warped_fmaps,_ = DatasetTools._get_warped_sampled(rgb_fmaps, gt_matrix)                        
        total_loss_backbone , detailed_batch_losses_backbone = loss_functions.get_losses_febackbone(warped_inputs,
                                                                                                    template_images,
                                                                                                    warped_fmaps,
                                                                                                    ir_fmaps,
                                                                                                    backbone_loss_function)
        # loss shouldn't be nan
        # assert tf.reduce_all(tf.math.is_finite(total_loss_backbone)), "total_loss_backbone Loss is NaN"    
        # add losses to epoch losses
        detailed_batch_losses_backbone = {key: value.numpy() for key, value in detailed_batch_losses_backbone.items()}
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
    # print(f"[train_loss] : {log}")
    return model , epochs_losses_summary , log


# test step for the regression head
def test_step(model,
                backBone,
                dataloader,
                predicting_homography,
                backbone_loss_function,
                loss_function_to_use): 
    """
    Test step for the regression head
    """
    epochs_losses_summary= {"backbone": defaultdict(list),
                            "regression_head": defaultdict(list)
                            }
    
    assert backBone is not None, "the feature embedding backbone is not defined"
    print(f"[INFO] testing  on {len(dataloader)} pairs")
    
    for i, batch in tqdm(enumerate(dataloader)):
        input_images, template_images, labels,_instances = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        # if len(input_images.shape) != 4:
        #     input_images = tf.expand_dims(input_images, axis=0)
        #     template_images  = tf.expand_dims(template_images, axis=0)
        #     labels = tf.expand_dims(labels, axis=0)
        
        # pass the input images through the backbone
        
        
        gt_matrix=DatasetTools.get_ground_truth_homographies(labels)
        warped_inputs, _ = DatasetTools._get_warped_sampled(input_images, gt_matrix)
        
        rgb_fmaps , ir_fmaps = backBone.call((input_images, template_images),training=False)
        
        # padd the ir_fmaps to match the shape of the rgb_fmaps
        
        # weighted sum if needed 
        summed_rgb_fmaps = tf.constant(1.0) * tf.cast(input_images,"float")   + tf.constant(0.2) * tf.cast(rgb_fmaps,"float")
        summed_ir_fmaps =  tf.constant(0.2) * tf.cast(template_images,"float") + tf.constant(1.0) *  tf.cast(ir_fmaps,'float')
        # padd the ir_fmaps to match the shape of the rgb_fmaps
        ir_fmaps_padded = backboneUtils.get_padded_fmaps(fmaps=summed_ir_fmaps, desired_shape = rgb_fmaps.shape)

        
        # concatenate the rgb_fmaps and ir_fmaps
        concatenated_fmaps = tf.concat([summed_rgb_fmaps, ir_fmaps_padded], axis=-1)
        

        predictions = model.call((concatenated_fmaps),training=False)
        total_loss , detailed_batch_losses = loss_functions.get_losses_regression_head( predictions = predictions, 
                                                                                        ground_truth_corners = labels,
                                                                                        gt_matrix = gt_matrix, 
                                                                                        predicting_homography=predicting_homography,
                                                                                        loss_function_to_use = loss_function_to_use)
        # assert tf.reduce_all(tf.math.is_finite(total_loss)), "Loss is NaN or Inf"
        # add losses to epoch losses
        detailed_batch_losses = {key: value.numpy() for key, value in detailed_batch_losses.items()}
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary["regression_head"][key].append(value)
            
        # backbone losses 
        warped_fmaps,_ = DatasetTools._get_warped_sampled(rgb_fmaps, gt_matrix)                        
        total_loss_backbone , detailed_batch_losses_backbone = loss_functions.get_losses_febackbone(warped_inputs,
                                                                                                    template_images,
                                                                                                    warped_fmaps,
                                                                                                    ir_fmaps,
                                                                                                    backbone_loss_function)
        # loss shouldn't be nan
        # assert tf.reduce_all(tf.math.is_finite(total_loss_backbone)), "BackBone Loss is NaN or Inf"    
        # add losses to epoch losses
        detailed_batch_losses_backbone = {key: value.numpy() for key, value in detailed_batch_losses_backbone.items()}
        for key, value in detailed_batch_losses_backbone.items():
            epochs_losses_summary["backbone"][key].append(value)
    
    # compute mean of losses
    for step, losses in epochs_losses_summary.items():
        for key, value in losses.items():
            epochs_losses_summary[step][key] = np.mean(value)

    # display losses
    log = " | ".join([str(str(i)+ " : " + str(k)) for i,k in epochs_losses_summary["regression_head"].items()])
    # print(f"[test_loss] : {log}")
    return epochs_losses_summary , log