import tensorflow as tf
import Tools.backboneUtils as BackBoneUtils
import Tools.utilities as common_utils
import Tools.datasetTools as DatasetTools
import numpy as np


def get_right_shapes(img_1, img_2):
    """
    Given two images, return the right shapes for them
    """
    
    img_1= tf.cast(img_1, dtype="float")
    img_2 = tf.cast(img_2, dtype="float")

    max_height = tf.math.maximum(img_1.shape[1], img_2.shape[1])
    max_width = tf.math.maximum(img_1.shape[2], img_2.shape[2])

    # img_1= tf.cast(img_1[:,:,:,:3], dtype="float")
    # img_2 = tf.cast(img_2[:,:,:,:3], dtype="float")

    pad_height1 = (max_height - img_1.shape[1]) // 2
    pad_width1 = (max_width - img_1.shape[2]) // 2
    paddings = [[0,0], [pad_height1, pad_height1], [pad_width1, pad_width1], [0,0]]
    img_1 = tf.pad(img_1, paddings=paddings, mode="CONSTANT", constant_values=0.0)

    pad_height2 = (max_height - img_2.shape[1]) // 2
    pad_width2 = (max_width - img_2.shape[2]) // 2
    paddings = [[0,0], [pad_height2, pad_height2], [pad_width2, pad_width2], [0,0]]
    img_2 = tf.pad(img_2, paddings=paddings, mode="CONSTANT", constant_values=0.0)
    
    # alternative
    # img1 = tf.image.resize_with_crop_or_pad(img_1, max_height, max_width)
    # img2 = tf.image.resize_with_crop_or_pad(img_2, max_height, max_width)
    
    return img_1, img_2
    

def mse_pixel(img_1, img_2):
    
    img_1, img_2 = get_right_shapes(img_1, img_2)
    
    img_1= tf.cast(img_1, dtype="float")
    img_2 = tf.cast(img_2, dtype="float")
    
    squared_diff = tf.math.square(img_1-img_2)
    # return tf.reduce_mean(tf.boolean_mask(squared_diff, tf.math.is_finite(squared_diff)))
    
    return tf.cast(tf.reduce_mean(squared_diff), dtype="float")

def mae_pixel(img_1, img_2):
    
    img_1, img_2 = get_right_shapes(img_1, img_2)
    img_1= tf.cast(img_1, dtype="float")
    img_2 = tf.cast(img_2, dtype="float")
    
    abs_diff = tf.math.abs(img_1-img_2)
    # return tf.reduce_mean(tf.boolean_mask(abs_diff, tf.math.is_finite(abs_diff)))
    return tf.cast(tf.reduce_mean(abs_diff), dtype="float")

def sse_pixel(img_1, img_2):
    
    img_1, img_2 = get_right_shapes(img_1, img_2)
    img_1= tf.cast(img_1, dtype="float")
    img_2 = tf.cast(img_2, dtype="float")
    
    squared_diff = tf.math.square(img_1-img_2)
    # return tf.reduce_sum(tf.boolean_mask(squared_diff, tf.math.is_finite(squared_diff)))
    return tf.cast(tf.reduce_sum(squared_diff), dtype="float")

def ssim_pixel(img_1, img_2):
        """
            Needs checking
        """
        img_1, img_2 = get_right_shapes(img_1, img_2)
        img_1= tf.cast(img_1, dtype="float")
        img_2 = tf.cast(img_2, dtype="float")
        
        ssim = tf.cast(tf.reduce_mean(tf.image.ssim(img_1, img_2, max_val=1.0)), dtype="float")
        
        # return tf.reduce_mean(tf.boolean_mask(ssim, tf.math.is_finite(ssim)))
        return tf.cast( tf.constant(1.0) - ssim , dtype="float")
    
    
    


    
    

    
@tf.function
def get_losses_febackbone(warped_inputs,
                            template_images,
                            warped_fmaps,
                            ir_fmaps,
                            loss_function):
    
    
        """
        Given feature maps, and images compute losses and return them in dictionary
        
        Args:
            loss_function: string : can be ["mse_pixel", "mae_pixel", "sse_pixel", "ssim_pixel"]
        
        Return:
                total_loss, detailed_losses
        """
        
        # loss functions dictionary dictionary
        loss_functions = {
                        "mse_pixel": mse_pixel,
                        "mae_pixel": mae_pixel,
                        "sse_pixel": sse_pixel,
                        "ssim_pixel": ssim_pixel
                        }
        
        loss_fn = loss_functions[loss_function]
        
        
        
        _fir_frgb  = loss_fn(template_images, warped_fmaps)#should be minimal
        _fir_Iir   = loss_fn(ir_fmaps,template_images)
        _frgb_Irgb = loss_fn(warped_fmaps,warped_inputs)#should be minimal
        _fir_Irgb  = loss_fn(ir_fmaps,warped_inputs)#should be minimal
        _frgb_Iir  = loss_fn(warped_fmaps,template_images)#
        _Iir_Irgb  = loss_fn(template_images,warped_inputs)

        
    
        total_loss_mse = tf.constant(0.0)
        losses_weights = tf.constant([1,.001,1,1,.0000001,.0000001], dtype="float")
        losses = tf.convert_to_tensor([_fir_frgb, _fir_Iir, _frgb_Irgb, _fir_Irgb, _frgb_Iir, _Iir_Irgb], dtype="float")
        # losses = [_fir_frgb, _fir_Iir, _frgb_Irgb, _fir_Irgb, _frgb_Iir, _Iir_Irgb]
        losses = tf.math.multiply(losses , losses_weights)
        # total_loss_mse = tf.reduce_sum(tf.boolean_mask(losses, tf.math.is_finite(losses)))
        total_loss_mse = tf.reduce_sum(losses)
        #add epsilon safe guard
        # total_loss_mse = tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + (total_loss_mse if tf.math.is_finite(total_loss_mse) else 0)
        total_loss_mse = tf.reduce_sum(tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + tf.cast((total_loss_mse if tf.math.is_finite(total_loss_mse) else 0.0) , dtype="float"))
        # create losss dictionary
        
        detailed_batch_losses = {"fir_frgb": _fir_frgb,
                                    "fir_Iir": _fir_Iir,
                                    "frgb_Irgb": _frgb_Irgb,
                                    "fir_Irgb": _fir_Irgb,
                                    "frgb_Iir": _frgb_Iir,
                                    "ir_Irgb": _Iir_Irgb,
                                    "total_loss": total_loss_mse
                                    }

        return total_loss_mse , detailed_batch_losses
    
@tf.function    
def get_losses_regression_head(predictions, 
                                ground_truth_corners,
                                gt_matrix , 
                                predicting_homography,
                                loss_function_to_use):
    """ 
    Given predicted matrix and ground truth matrix compute losses and return them in dictionary
    
    Args:
        predictions: [batch_size, 8] tensor can be either corners or homographies
        ground_truth_corners: [batch_size, 8] tensor 
        gt_matrix: [batch_size, 3,3] tensor 
        predictions_are_homographies: bool
    """
    losses = tf.constant(0.0)
    # if we are predicting homographies
    if predicting_homography:
        # append ones to to the predictions 
        ones = tf.cast(tf.ones((predictions.shape[0],1)),dtype="float")
        predictions = tf.concat([tf.cast(predictions,dtype="float"), ones], axis=-1)
        # reshape predictions to be [batch_size, 3,3]
        prediction_matrices = tf.reshape(predictions, (-1,3,3))
        # get predicted corners from predicted homographies
        prediction_corners = DatasetTools.homographies_to_corners(prediction_matrices)
        
    else:
        # corners are already predicted
        prediction_corners = predictions
        # convert corners to homographies
        with tf.device("/cpu:0"):
            prediction_matrices = DatasetTools.corners_to_homographies(prediction_corners)
    

    var1_intermidiate = tf.math.abs(prediction_matrices - gt_matrix)
    l1_homography_loss = tf.reduce_mean(tf.boolean_mask(var1_intermidiate, tf.math.is_finite(var1_intermidiate)))
    
    #
    var2_intermidiate = tf.math.square(prediction_matrices - gt_matrix)
    l2_homography_loss = tf.reduce_mean(tf.boolean_mask(var2_intermidiate, tf.math.is_finite(var2_intermidiate)))
    
    #
    var3_intermidiate = tf.math.abs(prediction_corners - ground_truth_corners)
    l1_corners_loss = tf.reduce_mean(tf.boolean_mask(var3_intermidiate, tf.math.is_finite(var3_intermidiate)))
    
    #
    var4_intermidiate = tf.math.square(prediction_corners - ground_truth_corners)
    l2_corners_loss = tf.reduce_mean(tf.boolean_mask(var4_intermidiate, tf.math.is_finite(var4_intermidiate)))
    
    # loss functions map
    loss_functions_to_weights = {   
                                    "l1_homography_loss": tf.constant([1.0, .0, .0, .0], dtype="float"),
                                    "l2_homography_loss": tf.constant([.0, 1.0, .0, .0], dtype="float"),
                                    "l1_corners_loss": tf.constant([.0, .0, 1.0, .0], dtype="float"),
                                    "l2_corners_loss": tf.constant([.0, .0, .0, 1.0], dtype="float")                                    
                                    }
    
    losses_weights = loss_functions_to_weights[loss_function_to_use]

    losses = tf.convert_to_tensor([l1_homography_loss, l2_homography_loss, l1_corners_loss, l2_corners_loss], dtype="float")
    # losses = [l1_homography_loss, l2_homography_loss, l1_corners_loss, l2_corners_loss]
    losses = tf.math.multiply(losses ,losses_weights)
    total_loss = tf.reduce_sum(tf.boolean_mask(losses, tf.math.is_finite(losses)))
    total_loss = tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + tf.cast((total_loss if tf.math.is_finite(total_loss) else 0.0),dtype="float")

    # create losss dictionary
    detailed_batch_losses = {"l1_homography_loss": l1_homography_loss,
                            "l2_homography_loss": l2_homography_loss,
                            "l1_corners_loss": l1_corners_loss,
                            "l2_corners_loss": l2_corners_loss,
                            "total_loss": total_loss
                            }
    # total loss must be a tensor
    # assert total_loss is not None, f"{losses}[ERROR] total_loss is None"
        
    return total_loss, detailed_batch_losses
    

    