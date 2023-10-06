import tensorflow as tf
import Tools.backboneUtils as BackBoneUtils
import Tools.utilities as common_utils


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
    
    
    fmaps_have_nans = common_utils.tensor_has_nan(img_2) or common_utils.tensor_has_nan(img_1) 
    assert (len(img_1.shape)==4 and len(img_2.shape)==4 ), "[ERROR] images must have 4 dimensions"
    assert not common_utils.tensor_has_nan(img_1) , "[ERROR] images must not have nans"
    assert not common_utils.tensor_has_nan(img_2) , "[ERROR] images must not have nans"
    
    
    img_1 = BackBoneUtils.get_fmaps_in_suitable_shape(img_1)
    img_2 = BackBoneUtils.get_fmaps_in_suitable_shape(img_2)
    
    img_1 = BackBoneUtils.get_fused_fmaps(img_1)
    img_2 = BackBoneUtils.get_fused_fmaps(img_2)
    
    assert img_1.shape == img_2.shape, "[ERROR] images must have the same shape"
    before_loss_nans = common_utils.tensor_has_nan(img_2) or common_utils.tensor_has_nan(img_1)
    assert not before_loss_nans , "[ERROR] loss cant be computed on  nans"
    
    # we can now compute loss 
    return tf.math.reduce_mean(tf.math.square(img_1-img_2))
    
    

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
    


def get_losses_febackbone(warped_inputs,template_images,warped_fmaps,ir_fmaps):
        """
        Given feature maps, and images compute losses and return them in dictionary
        
        Return:
                total_loss, detailed_losses
        """
        # compute similarity losses                
        _fir_frgb  = compute_similarity_differences_mse(template_images, warped_fmaps)#should be minimal
        _fir_Iir   = compute_similarity_differences_mse(ir_fmaps,template_images)
        _frgb_Irgb = compute_similarity_differences_mse(warped_fmaps,warped_inputs)#should be minimal
        _fir_Irgb  = compute_similarity_differences_mse(ir_fmaps,warped_inputs)#should be minimal
        _frgb_Iir  = compute_similarity_differences_mse(warped_fmaps,template_images)#
        _Iir_Irgb  = compute_similarity_differences_mse(template_images,warped_inputs)
        
        losses_weights = [1,.001,1,1,.0000001,.0000001]
        losses = [_fir_frgb, _fir_Iir, _frgb_Irgb, _fir_Irgb, _frgb_Iir, _Iir_Irgb]
        losses = [i*j for i,j in zip(losses,losses_weights)]
        total_loss_mse = tf.math.reduce_sum(losses)
        
        
        # create losss dictionary
        detailed_batch_losses = {"fir_frgb": _fir_frgb.numpy(),
                                    "fir_Iir": _fir_Iir.numpy(),
                                    "frgb_Irgb": _frgb_Irgb.numpy(),
                                    "fir_Irgb": _fir_Irgb.numpy(),
                                    "frgb_Iir": _frgb_Iir.numpy(),
                                    "ir_Irgb": _Iir_Irgb.numpy(),
                                    "total_loss": total_loss_mse.numpy()
                                    }
        
        return total_loss_mse , detailed_batch_losses