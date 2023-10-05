"""
will contain training functionality
"""
import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

from tqdm.auto import tqdm    

import Tools.backboneUtils as backboneUtils
import Tools.loss_functions as loss_functions
import Tools.datasetTools as DatasetTools


def train_step_first_stage(model,
                        dataloader,
                        optimizer):   
    model.compile(optimizer=optimizer) 
    epoch_losses_dict= {
                        "fir_frgb": [],
                        "fir_Iir": [],
                        "frgb_Irgb": [],
                        "fir_Irgb": [],
                        "frgb_Iir": [],
                        "ir_Irgb": [],
                        "total_loss": []
                    }
    for i, batch in enumerate(dataloader.take(8)):
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
                
                # compute similarity losses                
                _fir_frgb = loss_functions.compute_similarity_differences_mse(template_images, warped_fmaps)#should be minimal
                _fir_Iir = loss_functions.compute_similarity_differences_mse(ir_fmaps,template_images)
                _frgb_Irgb = loss_functions.compute_similarity_differences_mse(warped_fmaps,warped_inputs)#should be minimal
                _fir_Irgb = loss_functions.compute_similarity_differences_mse(ir_fmaps,warped_inputs)#should be minimal
                _frgb_Iir = loss_functions.compute_similarity_differences_mse(warped_fmaps,template_images)#
                _Iir_Irgb = loss_functions.compute_similarity_differences_mse(template_images,warped_inputs)
                
                losses_weights = [1,.001,1,1,.0000001,.0000001]
                losses = [_fir_frgb, _fir_Iir, _frgb_Irgb, _fir_Irgb, _frgb_Iir, _Iir_Irgb]
                losses = [i*j for i,j in zip(losses,losses_weights)]
                total_loss = tf.math.reduce_sum(losses)
                
                # loss shouldn't be nan
                assert not np.isnan(total_loss.numpy()), "Loss is NaN"
                
        all_parameters= model.trainable_variables
        grads = tape.gradient(total_loss, all_parameters)
        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))
        
        # create losss dictionary
        batch_losses = {"fir_frgb": _fir_frgb.numpy(),
                        "fir_Iir": _fir_Iir.numpy(),
                        "frgb_Irgb": _frgb_Irgb.numpy(),
                        "fir_Irgb": _fir_Irgb.numpy(),
                        "frgb_Iir": _frgb_Iir.numpy(),
                        "ir_Irgb": _Iir_Irgb.numpy(),
                        "total_loss": total_loss.numpy()}
        # add losses to epoch losses
        for key, value in batch_losses.items():
            epoch_losses_dict[key].append(value)
            
        # log = " ".join([str(i + " :" + k + "\n") for i,k in batch_losses.items()])
        # print(log)
    
    # compute mean of losses
    for key, value in epoch_losses_dict.items():
        epoch_losses_dict[key] = np.mean(value)
    
    # display losses
    log = " ".join([str(i + " :" + k + "\n") for i,k in epoch_losses_dict.items()])
    print(log)
    
    return epoch_losses_dict

def test_step__first_stage(model,
                        dataloader):
    test_results = [None]
    for i, batch in enumerate(dataloader.take(8)):
        pass
    return test_results
        

def train_first_stage(model: tf.keras.Model,
                    train_dataloader,
                    test_dataloader,
                    optimizer,
                    epochs):

    # 2. Create empty results dictionary
    print(f"[INFO] Training{model.name} for {epochs} epochs")
    results = {"train_loss": [],
                "test_results":[]}
    # 3. Loop over the epochs
    for  epoch in tqdm(range(epochs)):
        _per_epoch_train_losses = train_step_first_stage(model=model,
                                                dataloader=train_dataloader,
                                                optimizer=optimizer)
        
        
        _per_epoch_test_results = test_step__first_stage(model=model,
                                                    dataloader=test_dataloader)
        

        # 5. Update results dictionary
        results["test_results"].append(_per_epoch_test_results)
        results["train_loss"].append(_per_epoch_train_losses)


    return results 

