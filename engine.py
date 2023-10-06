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
import Tools.utilities as common_utils
from collections import defaultdict




def train_step_first_stage(model,
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

def test_step__first_stage(model,
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

def train_first_stage(model: tf.keras.Model,
                    train_dataloader,
                    test_dataloader,
                    optimizer,
                    epochs=1,
                    save_path="models",
                    save_as=f"featureEmbeddingBackBone",
                    save_frequency=1,
                    save_hard_frequency=50):
    
    # create a tag for the training
    log_tag = {
                    "model.name": model.name,
                    "epochs": epochs,
                    "train_size": len(train_dataloader),
                    "test_size": len(test_dataloader),
                    "tag_name": f"train_{model.name}_first_stage_{epochs}_epochs",
                    "per_epoch_metrics":{"train_loss": defaultdict(list),
                                            "test_results":defaultdict(list)
                                            },
                    "training_time": 0     
                }
    # 2. Create empty results dictionary
    print(f"[INFO] Training{model.name} for {epochs} epochs")

    from timeit import default_timer as timer
    start_time = timer()
    
    # 3. Loop over the epochs
    for  epoch in range(epochs):
        print(f"[INFO] Epoch {epoch+1}/{epochs}")
        _per_epoch_train_losses = train_step_first_stage(model=model,
                                                dataloader=train_dataloader,
                                                optimizer=optimizer)
        
        
        _per_epoch_test_results = test_step__first_stage(model=model,
                                                    dataloader=test_dataloader)
        # 6. Save model
        if (epoch+1) % save_frequency == 0:
            hard_tag = str(int((epoch+1)/save_hard_frequency) + 1)
            common_utils.save_model_weights(model=model,
                                            save_path=save_path,
                                            save_as=save_as,
                                            tag=str(hard_tag))
            
    
        # could be functionalized
        # 8.  Update logs dictionary
        for k,v in _per_epoch_train_losses.items():
            log_tag["per_epoch_metrics"]["train_loss"][k].append(v)
        for k,v in _per_epoch_test_results.items():
            log_tag["per_epoch_metrics"]["test_results"][k].append(v)
        
        end_time = timer()
        training_time = f"{(end_time-start_time)/3600:.2f} hours"
        log_tag["training_time"] = training_time

        # 7. Save results
        common_utils.save_logs(logs=log_tag,
                                save_path="logs",
                                save_as=f"{log_tag['tag_name']}.json")
                                
    return log_tag["per_epoch_metrics"] 
