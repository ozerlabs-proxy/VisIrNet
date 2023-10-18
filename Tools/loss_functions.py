import tensorflow as tf
import Tools.backboneUtils as BackBoneUtils
import Tools.utilities as common_utils
import Tools.datasetTools as DatasetTools
import numpy as np



def compute_similarity_differences_mse (img_1, img_2):
    # chanels_kept = tf.math.minimum(img_1.shape[3], img_2.shape[3])
    #just get the first 3 channels for now
    
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

    # img1 = tf.image.resize_with_crop_or_pad(img_1, max_height, max_width)
    # img2 = tf.image.resize_with_crop_or_pad(img_2, max_height, max_width)
    
    squared_diff = tf.math.square(img_1-img_2)
    # return tf.reduce_mean(tf.boolean_mask(squared_diff, tf.math.is_finite(squared_diff)))
    return tf.reduce_mean(squared_diff)
    
    
    
def compute_similarity_differences_mse_old(img_1, img_2):
        
    """
        Compute different loss for backbone
    """

    tf.config.run_functions_eagerly(True)
    img_1 = BackBoneUtils.get_fmaps_in_suitable_shape(img_1)
    img_2 = BackBoneUtils.get_fmaps_in_suitable_shape(img_2)
    before_loss_nans = common_utils.tensor_has_nan(img_2) or common_utils.tensor_has_nan(img_1)
    
 
    img_1 = BackBoneUtils.get_fused_fmaps(img_1)
    img_2 = BackBoneUtils.get_fused_fmaps(img_2)
    before_loss_nans = common_utils.tensor_has_nan(img_2) or common_utils.tensor_has_nan(img_1)

    squared_diff = tf.math.square(img_1-img_2)
    return tf.reduce_mean(tf.boolean_mask(squared_diff, tf.math.is_finite(squared_diff)))
    
    
def combine_ssim_losses(ssim_fir_frgb,
                        ssim_fir_Iir,
                        ssim_frgb_Irgb,
                        ssim_fir_Irgb,
                        ssim_frgb_Iir,
                        ssim_Iir_Irgb):
    margin_factor=0.000001                
    margin_ir= margin_factor * ssim_fir_Iir
    total_loss= ssim_frgb_Irgb + 0.5*(ssim_fir_frgb  + ssim_fir_Irgb) + margin_ir
    return total_loss

@tf.function
def get_losses_febackbone(warped_inputs,
                            template_images,
                            warped_fmaps,
                            ir_fmaps):
        """
        Given feature maps, and images compute losses and return them in dictionary
        
        Return:
                total_loss, detailed_losses
        """
        
        # check for assumptions
        # create copies of inputs
        # warped_inputs = tf.identity(warped_inputs)
        # template_images = tf.identity(template_images)
        # warped_fmaps = tf.identity(warped_fmaps)
        # ir_fmaps = tf.identity(ir_fmaps)
        
        
        
        
        # assert tf.reduce_all(tf.math.is_finite(warped_inputs)), "[ERROR] warped_inputs has nans"
        # assert tf.reduce_all(tf.math.is_finite(template_images)), "[ERROR] template_images has nans"
        # assert tf.reduce_all(tf.math.is_finite(warped_fmaps)), "[ERROR] warped_fmaps has nans"
        # assert tf.reduce_all(tf.math.is_finite(ir_fmaps)), "[ERROR] ir_fmaps has nans"
        
        # compute similarity losses                
        _fir_frgb  = compute_similarity_differences_mse(template_images, warped_fmaps)#should be minimal
        _fir_Iir   = compute_similarity_differences_mse(ir_fmaps,template_images)
        _frgb_Irgb = compute_similarity_differences_mse(warped_fmaps,warped_inputs)#should be minimal
        _fir_Irgb  = compute_similarity_differences_mse(ir_fmaps,warped_inputs)#should be minimal
        _frgb_Iir  = compute_similarity_differences_mse(warped_fmaps,template_images)#
        _Iir_Irgb  = compute_similarity_differences_mse(template_images,warped_inputs)
        
    
        total_loss_mse = tf.constant(0.0)
        losses_weights = tf.constant([1,.001,1,1,.0000001,.0000001],dtype="float")
        losses = tf.convert_to_tensor([_fir_frgb, _fir_Iir, _frgb_Irgb, _fir_Irgb, _frgb_Iir, _Iir_Irgb], dtype="float")
        # losses = [_fir_frgb, _fir_Iir, _frgb_Irgb, _fir_Irgb, _frgb_Iir, _Iir_Irgb]
        losses = tf.math.multiply(losses , losses_weights)
        total_loss_mse = tf.reduce_sum(tf.boolean_mask(losses, tf.math.is_finite(losses)))
        #add epsilon safe guard
        # total_loss_mse = tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + (total_loss_mse if tf.math.is_finite(total_loss_mse) else 0)
        total_loss_mse = tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + tf.cast((total_loss_mse if tf.math.is_finite(total_loss_mse) else 0.0),dtype="float")
        # create losss dictionary
        
        detailed_batch_losses = {"fir_frgb": _fir_frgb,
                                    "fir_Iir": _fir_Iir,
                                    "frgb_Irgb": _frgb_Irgb,
                                    "fir_Irgb": _fir_Irgb,
                                    "frgb_Iir": _frgb_Iir,
                                    "ir_Irgb": _Iir_Irgb,
                                    "total_loss": total_loss_mse
                                    }
        # # loss shouldn't be nan
        # with tf.init_scope():
        #     assert tf.reduce_all(tf.math.is_finite(total_loss_mse)), f"{detailed_batch_losses}[ERROR] total_loss is None"
        return total_loss_mse , detailed_batch_losses
    
    
def get_losses_regression_head(predictions, 
                                ground_truth_corners,
                                gt_matrix , 
                                predicting_homography):
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
        prediction_matrices = DatasetTools.corners_to_homographies(prediction_corners)
    
    # assertions
    # assert tf.reduce_all(tf.math.is_finite(prediction_matrices)) , "[ERROR] prediction_matrices has nans"
    # assert tf.reduce_all(tf.math.is_finite(prediction_corners)) , "[ERROR] prediction_corners has nans"
    # assert tf.reduce_all(tf.math.is_finite(gt_matrix)) , "[ERROR] gt_matrix has nans"
    # assert tf.reduce_all(tf.math.is_finite(ground_truth_corners)) , "[ERROR] ground_truth_corners has nans"
    

    
    #tf.reduce_mean(tf.boolean_mask(squared_diff, tf.math.is_finite(squared_diff)))
    # compute the loss
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
    
    
    losses_weights = tf.constant([.0,1.0,.0,.0] if predicting_homography else [.0,.0,.0,1.0], dtype="float")
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
    

    