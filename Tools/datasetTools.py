import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pathlib import Path
import numpy as np
import PIL.Image as Image
import json
import cv2
from Tools.warper import Warper



def _get_warped_sampled(images, homography_matrices, source_shape=(128,128)):
    """
        check if the transformed images have nans
    """
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    height_template, width_template = source_shape
    batch_size = images.shape[0]

    
    _warper = Warper(batch_size,height_template=height_template,width_template=width_template)
    warped_sampled = _warper.projective_inverse_warp(images, homography_matrices)
    warped_sampled = tf.cast(warped_sampled, tf.float32)
    
    _transformed_images_have_nans = tf.math.is_nan(warped_sampled).numpy().any()  
    
    return warped_sampled, _transformed_images_have_nans  



def _transformed_images(images, homography_matrices):
    """
        check if the transformed images have nans
    """
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        
    batch_size = images.shape[0]
    _warper = Warper(batch_size,height_template=128,width_template=128)
    warped_sampled = _warper.projective_inverse_warp(images, homography_matrices)
    
    _transformed_images_have_nans =tf.math.is_nan(warped_sampled).numpy().any()  
    
    return warped_sampled, _transformed_images_have_nans  


def get_initial_motion_matrix():
    """

    """
    src_points = [[0, 0], [127, 0],[0, 127], [127, 127]]
    tgt_points = [[32,32],[159,32],[32,159],[159,159]]

    src_points = np.reshape(src_points, [4, 1, 2])
    tgt_points = np.reshape(tgt_points, [4, 1, 2])
    # find homography
    h_matrix, _ = cv2.findHomography(src_points.astype(np.float32), 
                                            tgt_points.astype(np.float32), 
                                            0)
    matrix_size = h_matrix.size
    if matrix_size == 0:
        h_matrix = np.zeros([3, 3]) * np.nan
    h_matrix = h_matrix / h_matrix[2,2]    
    return np.asarray(h_matrix).astype(np.float32)

def get_ground_truth_homographies(u_v_list):
    """
        u_v_list: [batch_size, 8]
        return: [batch_size, 3, 3]
        homography matrices from u_v_list
    """

    batch_size = u_v_list.shape[0]
    u_list = u_v_list[:, :4]
    v_list = u_v_list[:, 4:]
    _initial_motion_matrix = get_initial_motion_matrix()
    matrix_list = []
    for i in range(batch_size):
        # src_points = [[0, 0], [127, 0],[0, 127], [127, 127]]
        src_points = [[32,32],[159,32],[32,159],[159,159]]
        
        tgt_points = np.concatenate([u_list[i:(i + 1), :], v_list[i:(i + 1), :]], axis=0)
        tgt_points = np.transpose(tgt_points)
        tgt_points = np.expand_dims(tgt_points, axis=1)

        src_points = np.reshape(src_points, [4, 1, 2])
        tgt_points = np.reshape(tgt_points, [4, 1, 2])
        # find homography
        h_matrix, _ = cv2.findHomography(src_points.astype(np.float32), 
                                                tgt_points.astype(np.float32), 
                                                0)
        matrix_size = h_matrix.size
        if matrix_size == 0:
            h_matrix = np.zeros([3, 3]) * np.nan
        matrix_list.append(h_matrix)
        
    homography_matrices = np.asarray(matrix_list).astype(np.float32)
    homography_matrices = tf.matmul(homography_matrices,_initial_motion_matrix) 
    dividend = homography_matrices[...,2,2]
    dividend = tf.expand_dims(dividend, axis=-1)
    dividend = tf.expand_dims(dividend, axis=-1)
    homography_matrices = tf.divide(homography_matrices,dividend)
    assert not tf.math.is_nan(homography_matrices).numpy().any(), "homography matrices have nan values"
    return homography_matrices

def get_inverse_homographies(homography_matrices):
    """
        get the inverse homographies if they exist,
        else return the nan matrices with the same shape
    """
    homography_matrices = tf.cast(homography_matrices, tf.float32)
    try:
        inverse_matrices = tf.linalg.inv(homography_matrices)
        inverse_matrices = inverse_matrices / inverse_matrices[...,2,2]
        return inverse_matrices
    except:
        return tf.fill(homography_matrices.shape, np.nan)

def is_invertible(homography_matrices):
    """ 
        get inverses and check if they are invertible
    """
    inverse_matrices = get_inverse_homographies(homography_matrices)
    invertible = not tf.math.is_nan(inverse_matrices).numpy().any()
    
    return inverse_matrices, invertible

def homographies_to_corners(prediction_matrices):
    """
    Given homography matrices compute the corners that initial corners would map to
    
    initial corners are [[0,0],[127,0],[0,127],[127,127]]
    
    Args:
        prediction_matrices: [batch_size, 3,3] tensor 
        
    Returns:
        new_four_points: [batch_size, 8] tensor u and v coordinates of the new corners
        
    """
    
    assert len(prediction_matrices.shape) == 3, "prediction_matrices should be [batch_size, 3,3]"
    batch_size = prediction_matrices.shape[0]
    # get initial corners
    initial_corners = np.asarray([[0,0,1],[127,0,1],[0,127,1],[127,127,1]])
    initial_corners = np.transpose(initial_corners)
    initial_corners = np.expand_dims(initial_corners, axis=0)
    initial_corners = np.tile(initial_corners, [batch_size, 1, 1]).astype(np.float32)
    initial_corners = tf.dtypes.cast(initial_corners, tf.float32)

    prediction_matrices = tf.dtypes.cast(prediction_matrices, tf.float32)
    new_four_points = tf.matmul(prediction_matrices, initial_corners)

    new_four_points_scale = new_four_points[:, 2:, :]
    new_four_points = new_four_points / new_four_points_scale

    new_four_points = new_four_points[:, :2, :]
    new_four_points = tf.reshape(new_four_points, (-1,8))
    
    return new_four_points
        

def corners_to_homographies(prediction_corners):
    """
    Given corners compute the homographies that would map the initial corners to the predicted corners
    
    Args:
        prediction_corners: [batch_size, 8] tensor u and v coordinates of the new corners
    
    """
    
    """
        u_v_list: [batch_size, 8]
        return: [batch_size, 3, 3]
        homography matrices from u_v_list
    """
    assert len(prediction_corners.shape) == 2, "prediction_corners should be [batch_size, 8]"
    batch_size = prediction_corners.shape[0]
    u_list = prediction_corners[:, :4]
    v_list = prediction_corners[:, 4:]
    
    matrix_list = []
    for i in range(batch_size):
        src_points = [[0, 0], [127, 0],[0, 127], [127, 127]]
        
        tgt_points = np.concatenate([u_list[i:(i + 1), :], v_list[i:(i + 1), :]], axis=0)
        tgt_points = np.transpose(tgt_points)
        tgt_points = np.expand_dims(tgt_points, axis=1)

        src_points = np.reshape(src_points, [4, 1, 2])
        tgt_points = np.reshape(tgt_points, [4, 1, 2])
        # find homography
        h_matrix, _ = cv2.findHomography(src_points.astype(np.float32), 
                                                tgt_points.astype(np.float32), 
                                                0)
        matrix_size = h_matrix.size
        if matrix_size == 0:
            h_matrix = np.zeros([3, 3]) * np.nan
        matrix_list.append(h_matrix)
        
    homography_matrices = np.asarray(matrix_list).astype(np.float32)
    dividend = homography_matrices[...,2,2]
    dividend = tf.expand_dims(dividend, axis=-1)
    dividend = tf.expand_dims(dividend, axis=-1)
    homography_matrices = tf.divide(homography_matrices,dividend)
    assert not tf.math.is_nan(homography_matrices).numpy().any(), "homography matrices have nan values"
    return homography_matrices