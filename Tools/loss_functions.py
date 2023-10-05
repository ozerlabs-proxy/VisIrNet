import tensorflow as tf
from Tools.backboneUtils import _similarity_loss
from  Tools.warper import Warper


def get_backbone_loss_ssim(rgb_images,
                                ir_images,
                                rgb_fmaps,
                                ir_fmaps,
                                gt_matrix):
    
    """
        Compute different loss for backbone
    """
    
    batch_size=rgb_images.shape[0]
    
    _warper = Warper(batch_size=batch_size,
                                    height_template=ir_fmaps.shape[1],
                                    width_template=ir_fmaps.shape[2]
                                    )
    frgb_warped_to_fir=_warper.projective_inverse_warp(rgb_fmaps, gt_matrix)
    Irgb_warped_to_Iir=_warper.projective_inverse_warp(rgb_images, gt_matrix)

    ssim_fir_frgb=tf.reduce_mean(_similarity_loss(ir_fmaps,frgb_warped_to_fir))#should be minimal
    ssim_fir_Iir=tf.reduce_mean(_similarity_loss(ir_fmaps,ir_images))#
    ssim_frgb_Irgb=tf.reduce_mean(_similarity_loss(frgb_warped_to_fir,Irgb_warped_to_Iir))#should be minimal
    ssim_fir_Irgb=tf.reduce_mean(_similarity_loss(ir_fmaps,Irgb_warped_to_Iir))#should be minimal
    ssim_frgb_Iir=tf.reduce_mean(_similarity_loss(frgb_warped_to_fir,ir_images))#
    ssim_Iir_Irgb=tf.reduce_mean(_similarity_loss(ir_images,Irgb_warped_to_Iir))#
    
    # losses = {  "ssim_fir_frgb":ssim_fir_frgb,
    #             "ssim_fir_Iir":ssim_fir_Iir,
    #             "ssim_frgb_Irgb":ssim_frgb_Irgb,
    #             "ssim_fir_Irgb":ssim_fir_Irgb,
    #             "ssim_frgb_Iir":ssim_frgb_Iir,
    #             "ssim_Iir_Irgb":ssim_Iir_Irgb
    #             }
    return ssim_fir_frgb,ssim_fir_Iir,ssim_frgb_Irgb,ssim_fir_Irgb,ssim_frgb_Iir,ssim_Iir_Irgb

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
    

