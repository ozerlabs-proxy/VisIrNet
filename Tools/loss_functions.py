import tensorflow as tf
import Tools.backboneUtils as BackBoneUtils
import Tools.utilities as common_utils
import Tools.datasetTools as DatasetTools
import numpy as np


def compute_similarity_differences_mse(img_1, img_2):
        
    """
        Compute different loss for backbone
    """
    # print(f":[DEBUGGING")
    # # print types and shapes
    # print(f":[DEBUGGING] img_1 type {type(img_1)}")
    # print(f":[DEBUGGING] img_2 type {type(img_2)}")
    # print(f":[DEBUGGING] img_1 shape {img_1.shape}")
    # print(f":[DEBUGGING] img_2 shape {img_2.shape}")
    
    
    # fmaps_have_nans = common_utils.tensor_has_nan(img_2) or common_utils.tensor_has_nan(img_1) 
    assert (len(img_1.shape)==4 and len(img_2.shape)==4 ), "[ERROR] images must have 4 dimensions"
    assert not common_utils.tensor_has_nan(img_1) , "[ERROR] images must not have nans"
    assert not common_utils.tensor_has_nan(img_2) , "[ERROR] images must not have nans"
    
    
    img_1 = BackBoneUtils.get_fmaps_in_suitable_shape(img_1)
    img_2 = BackBoneUtils.get_fmaps_in_suitable_shape(img_2)
    before_loss_nans = common_utils.tensor_has_nan(img_2) or common_utils.tensor_has_nan(img_1)
    assert not before_loss_nans , "[ERROR] loss cant be computed on  nans"
    
    img_1 = BackBoneUtils.get_fused_fmaps(img_1)
    img_2 = BackBoneUtils.get_fused_fmaps(img_2)
    
    assert img_1.shape == img_2.shape, "[ERROR] images must have the same shape"
    before_loss_nans = common_utils.tensor_has_nan(img_2) or common_utils.tensor_has_nan(img_1)
    assert not before_loss_nans , "[ERROR] loss cant be computed on  nans"
    
    # we can now compute loss 
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
        
        
        
        
        assert np.isfinite(warped_inputs).all(), "[ERROR] warped_inputs has nans"
        assert np.isfinite(template_images).all(), "[ERROR] template_images has nans"
        assert np.isfinite(warped_fmaps).all(), "[ERROR] warped_fmaps has nans"
        assert np.isfinite(ir_fmaps).all(), "[ERROR] ir_fmaps has nans"
        
        # compute similarity losses                
        _fir_frgb  = compute_similarity_differences_mse(template_images, warped_fmaps)#should be minimal
        _fir_Iir   = compute_similarity_differences_mse(ir_fmaps,template_images)
        _frgb_Irgb = compute_similarity_differences_mse(warped_fmaps,warped_inputs)#should be minimal
        _fir_Irgb  = compute_similarity_differences_mse(ir_fmaps,warped_inputs)#should be minimal
        _frgb_Iir  = compute_similarity_differences_mse(warped_fmaps,template_images)#
        _Iir_Irgb  = compute_similarity_differences_mse(template_images,warped_inputs)
        
    
        total_loss_mse = 0
        losses_weights = [1,.001,1,1,.0000001,.0000001]
        losses = [_fir_frgb, _fir_Iir, _frgb_Irgb, _fir_Irgb, _frgb_Iir, _Iir_Irgb]
        losses = [i*j for i,j in zip(losses,losses_weights)]
        total_loss_mse = tf.reduce_sum(tf.boolean_mask(losses, tf.math.is_finite(losses)))
        #add epsilon safe guard
        # total_loss_mse = tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + (total_loss_mse if tf.math.is_finite(total_loss_mse) else 0)
        total_loss_mse = tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + tf.cast((total_loss_mse if tf.math.is_finite(total_loss_mse) else 0),dtype="float")
        # create losss dictionary
        detailed_batch_losses = {"fir_frgb": _fir_frgb.numpy(),
                                    "fir_Iir": _fir_Iir.numpy(),
                                    "frgb_Irgb": _frgb_Irgb.numpy(),
                                    "fir_Irgb": _fir_Irgb.numpy(),
                                    "frgb_Iir": _frgb_Iir.numpy(),
                                    "ir_Irgb": _Iir_Irgb.numpy(),
                                    "total_loss": total_loss_mse.numpy()
                                    }
        # loss shouldn't be nan
        assert np.isfinite(total_loss_mse).all(), f"{detailed_batch_losses}[ERROR] total_loss is None"
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
    losses = None
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
    assert np.isfinite(prediction_matrices).all() , "[ERROR] prediction_matrices has nans"
    assert np.isfinite(prediction_corners).all() , "[ERROR] prediction_corners has nans"
    assert np.isfinite(gt_matrix).all() , "[ERROR] gt_matrix has nans"
    assert np.isfinite(ground_truth_corners).all() , "[ERROR] ground_truth_corners has nans"
    

    
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
    
    
    losses_weights = [.0,1.0,.0,.0] if predicting_homography else [.0,.0,.0,1.0]
    losses = [l1_homography_loss, l2_homography_loss, l1_corners_loss, l2_corners_loss]
    losses = [i*j for i,j in zip(losses,losses_weights)]
    total_loss = tf.reduce_sum(tf.boolean_mask(losses, tf.math.is_finite(losses)))
    total_loss = tf.convert_to_tensor(tf.keras.backend.epsilon(),dtype="float") + tf.cast((total_loss if tf.math.is_finite(total_loss) else 0),dtype="float")

    # create losss dictionary
    detailed_batch_losses = {"l1_homography_loss": l1_homography_loss.numpy(),
                            "l2_homography_loss": l2_homography_loss.numpy(),
                            "l1_corners_loss": l1_corners_loss.numpy(),
                            "l2_corners_loss": l2_corners_loss.numpy(),
                            "total_loss": total_loss.numpy()
                            }
    # total loss must be a tensor
    # assert total_loss is not None, f"{losses}[ERROR] total_loss is None"
        
    return total_loss, detailed_batch_losses
    

    