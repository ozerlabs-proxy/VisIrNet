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



def train_step_first_stage(model,
                        dataloader,
                        optimizer):   
    model.compile(optimizer=optimizer) 
    train_losses = []
    for i, batch in enumerate(dataloader.take(8)):
        input_image, template_image, label ,_instance = batch
        
        # add batch dim if shape is not (batch_size, height, width, channels)
        if len(input_image.shape) != 4:
            input_image = tf.expand_dims(input_image, axis=0)
            template_image  = tf.expand_dims(template_image, axis=0)
            label = tf.expand_dims(label, axis=0)
        
        gt_matrix=backboneUtils.gt_motion(label)
        # print(f"[INFO] gt_matrix shapes: {gt_matrix.shape}")
        # print(f"[INFO] input_image shapes: {input_image.shape}")
        # print(f"[INFO] template_image shapes: {template_image.shape}")
        # print(f"[INFO] gt_matrix shapes: {gt_matrix.shape}")
        
        with tf.GradientTape() as tape:
                rgb_feature_maps , ir_feature_maps=model.call((input_image, template_image))
                
                losses=loss_functions.get_backbone_loss_ssim(rgb_images=input_image,
                                                                ir_images=template_image,
                                                                rgb_fmaps=rgb_feature_maps,
                                                                ir_fmaps=ir_feature_maps,
                                                                gt_matrix=gt_matrix)
                ssim_fir_frgb,ssim_fir_Iir,ssim_frgb_Irgb,ssim_fir_Irgb,ssim_frgb_Iir,ssim_Iir_Irgb=losses
                total_loss= loss_functions.combine_ssim_losses(ssim_fir_frgb,
                                                                ssim_fir_Iir,
                                                                ssim_frgb_Irgb,
                                                                ssim_fir_Irgb,
                                                                ssim_frgb_Iir,
                                                                ssim_Iir_Irgb)

        all_parameters=model.trainable_variables
        grads = tape.gradient(total_loss, all_parameters)
        grads=[tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))
        
        log = f"{i} " \
                f"-totalLoss:{np.float64(total_loss)} " \
                f" -frgb_Irgb:{np.float64(ssim_frgb_Irgb)} " \
                f" -fir_Irgb:{np.float64(ssim_fir_Irgb)} " \
                f" -fir_frgb:{np.float64(ssim_fir_frgb)} " \
                f" -fir_Iir:{np.float64(ssim_fir_Iir)} " \
                f" -Iir_Irgb:{np.float64(ssim_Iir_Irgb)} " \

        # total_losses.append(np.float(total_loss))
        # fir_frgb.append(np.float(ssim_fir_frgb))
        # frgb_Irgb.append(np.float(ssim_frgb_Irgb))
        # fir_Iir.append(np.float(ssim_fir_Iir))
        # fir_Irgb.append(np.float(ssim_fir_Irgb))
        # Iir_Irgb.append(np.float(ssim_Iir_Irgb))

        train_losses.append(total_loss)
    # train_losses = [None]
    return train_losses

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
        train_losses = train_step_first_stage(model=model,
                                                dataloader=train_dataloader,
                                                optimizer=optimizer)
        
        
        test_results = test_step__first_stage(model=model,
                                                    dataloader=test_dataloader)
        

        # 5. Update results dictionary
        results["test_results"].append(test_results)
        results["train_loss"].append(train_losses)


    return results 

