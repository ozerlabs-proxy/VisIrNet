import numpy as np
import tensorflow as tf
import cv2


def gt_motion(u_v_list):
    batch_size=u_v_list.shape[0]
    
    u_list = u_v_list[:, :4]
    v_list = u_v_list[:, 4:]
    matrix_list = []
    for i in range(batch_size):
        src_points = [[0, 0], [127, 0], [127, 127], [0, 127]]
        
        tgt_points = np.concatenate([u_list[i:(i + 1), :], v_list[i:(i + 1), :]], axis=0)
        tgt_points = np.transpose(tgt_points)
        tgt_points = np.expand_dims(tgt_points, axis=1)

        src_points = np.reshape(src_points, [4, 1, 2])
        tgt_points = np.reshape(tgt_points, [4, 1, 2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points, 0)
        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)

def _similarity_loss(img_1, img_2):

    # #shape must be same
    # if img_1.shape!=img_2.shape:
    img_1 = get_fmaps_in_suitable_shape(img_1)
    img_2 = get_fmaps_in_suitable_shape(img_2)
    

    fmaps_1=get_fused_fmaps(img_1)
    fmaps_2=get_fused_fmaps(img_2)

    return tf.math.pow((fmaps_1 - fmaps_2), 2)

def get_fused_fmaps(feature_maps):
    initializer = tf.keras.initializers.Ones()
    input_shape = feature_maps.shape

    # max_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',trainable=False)
    # fmap_intermediate=max_pool_2d(feature_maps)
    fmap_intermediate=feature_maps

    fineFmaps = tf.keras.layers.Conv2D(1, 3, activation='linear', padding="same",use_bias=False,trainable=False, kernel_initializer=initializer,input_shape=input_shape[1:])(fmap_intermediate)
    # fineFmaps = tf.image.rgb_to_grayscale(feature_maps)
    return fineFmaps

def get_fmaps_in_suitable_shape(feature_maps):
    initializer = tf.keras.initializers.Ones()
    input_shape = feature_maps.shape
    max_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),trainable=False, strides=(1, 1), padding='same')
    y_intermediate=max_pool_2d(feature_maps)
    transformed_fmaps = tf.keras.layers.Conv2D(3, 3, activation='linear', padding="same",use_bias=False,trainable=False,kernel_initializer=initializer,input_shape=input_shape[1:])(y_intermediate)
    return transformed_fmaps
