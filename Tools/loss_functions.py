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
    
