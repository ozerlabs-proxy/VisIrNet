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
                    optimizer,
                    loss_function):  
    
    dataloader = dataloader.shuffle(1000)   
    # model.compile(optimizer=optimizer , experimental_run_tf_function=False) 
    # model.compile(optimizer=optimizer) 
    epochs_losses_summary= defaultdict(list)
    
    print(f"[INFO] training  on {len(dataloader)} pairs")

    for i, batch in tqdm(enumerate(dataloader.take(16))):

    
        input_images, template_images, labels,_instances = batch
        
        # tf.config.run_functions_eagerly(True)
        gt_matrix = DatasetTools.get_ground_truth_homographies(labels)
        # tf.config.run_functions_eagerly(False)
        
        warped_inputs, _ = DatasetTools._get_warped_sampled(images = input_images,  homography_matrices = gt_matrix)
        

        with tf.GradientTape() as tape:
                        #persistent=True
                        rgb_fmaps , ir_fmaps = model.call((input_images, template_images), training=True)
                        # total_loss = tf.constant(0.0)
                        tape.watch(rgb_fmaps)
                        tape.watch(ir_fmaps)
                        
                        
                        warped_fmaps,_ = DatasetTools._get_warped_sampled( images = rgb_fmaps, 
                                                                            homography_matrices = gt_matrix)
                        
                        tape.watch(warped_fmaps)
                        
                                        
                        total_loss , detailed_batch_losses = loss_functions.get_losses_febackbone( warped_inputs,
                                                                                                    template_images,
                                                                                                    warped_fmaps,
                                                                                                    ir_fmaps,
                                                                                                    loss_function)
                        
                        
                    
                    
        # get gradients and backpropagate                
        all_parameters= model.trainable_variables
        grads = tape.gradient(total_loss, all_parameters, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        grads_are_safe = np.array([ tf.math.is_finite(g).numpy().all() for g in grads ]).all()
        
        assert grads_are_safe, F"Gradients are not safe at iteration: {i}"
        #backpropagate

        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads] 
        optimizer.apply_gradients(zip(grads, all_parameters))
        
        detailed_batch_losses = {str(i): k.numpy() for i, k in detailed_batch_losses.items()}
        # loss_message = " | ".join([str(str(i)+ " : " + str(k)) for i,k in detailed_batch_losses.items()])
        
        # add losses to epoch losses
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary[key].append(value)

        
    # compute mean of losses
    for key, value in epochs_losses_summary.items():
        epochs_losses_summary[key] = np.mean(value)
    
    # display losses
    # display losses
    log = " | ".join([str(str(i)+ " : " + str(k)) for i,k in epochs_losses_summary.items()])

    # print(f"[train_loss] : {log}")

    return model, epochs_losses_summary ,log



# test step for the feature embedding backbone
def test_step(model,
                dataloader,
                loss_function
                ): 
    
    epochs_losses_summary= defaultdict(list)
    
    print(f"[INFO] testing  on {len(dataloader)} pairs")
    
    for i, batch in tqdm(enumerate(dataloader.take(16))):

        input_images, template_images, labels,_instances = batch

        gt_matrix = DatasetTools.get_ground_truth_homographies(labels)
        # assert gt_matrix.shape == (input_images.shape[0], 3, 3), "gt_matrix shape is not (batch_size, 3, 3)"
        
        warped_inputs, _ = DatasetTools._get_warped_sampled(images = input_images, 
                                                            homography_matrices = gt_matrix)
        
        rgb_fmaps , ir_fmaps = model.call((input_images, template_images), training=False)
        warped_fmaps, _  = DatasetTools._get_warped_sampled(images = rgb_fmaps, 
                                                            homography_matrices = gt_matrix)
        
        total_loss , detailed_batch_losses = loss_functions.get_losses_febackbone(warped_inputs,
                                                                                    template_images,
                                                                                    warped_fmaps,
                                                                                    ir_fmaps,
                                                                                    loss_function)
            
        
        # loss shouldn't be nan
        # assert tf.reduce_all(tf.math.is_finite(tf.cast(total_loss, dtype="float"))), "Loss is NaN"
        
        # add losses to epoch losses
        detailed_batch_losses = {str(i): k.numpy() for i, k in detailed_batch_losses.items()}
        for key, value in detailed_batch_losses.items():
            epochs_losses_summary[key].append(value)
            
        # log = " ".join([str(i + " :" + k + "\n") for i,k in batch_losses.items()])
        # print(log)
    
    # compute mean of losses
    for key, value in epochs_losses_summary.items():
        epochs_losses_summary[key] = np.mean(value)
    
    # display losses
    log = " | ".join([str(str(i)+ " :" + str(k)) for i,k in epochs_losses_summary.items()])
    # print(f"[test_loss] : {log}")
    
    return epochs_losses_summary ,log
