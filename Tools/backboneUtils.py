import numpy as np
import tensorflow as tf
import cv2



def get_fused_fmaps(feature_maps):
    tf.config.run_functions_eagerly(True)
    # with tf.init_scope():
        
    initializer = tf.keras.initializers.Ones()
    input_shape = feature_maps.shape

    fmap_intermediate = feature_maps

    fineFmaps = tf.keras.layers.Conv2D(1, 
                                        3, 
                                        activation='linear', 
                                        padding="same",
                                        use_bias=False,
                                        trainable=False, 
                                        kernel_initializer=initializer,
                                        input_shape=input_shape[1:])(fmap_intermediate)
    # fineFmaps = tf.image.rgb_to_grayscale(feature_maps)
    tf.config.run_functions_eagerly(False)
    return tf.convert_to_tensor(fineFmaps, dtype=tf.float32)

def get_fmaps_in_suitable_shape(feature_maps):
    tf.config.run_functions_eagerly(True)
    # with tf.init_scope():
    initializer = tf.keras.initializers.Ones()
    input_shape = feature_maps.shape
    max_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                    trainable=False, 
                                                    strides=(1, 1), 
                                                    padding='same')
    y_intermediate=max_pool_2d(feature_maps)
    transformed_fmaps = tf.keras.layers.Conv2D(3, 
                                                3, 
                                                activation='linear', 
                                                padding="same",
                                                use_bias=False,
                                                trainable=False,
                                                kernel_initializer=initializer,
                                                input_shape=input_shape[1:])(y_intermediate)
    tf.config.run_functions_eagerly(False)
    return tf.convert_to_tensor(transformed_fmaps, dtype=tf.float32)


def get_padded_fmaps(fmaps, desired_shape):
    """
        Convert feature maps from ir feb to have same size as rgb fmaps
        Args:
            fmaps: feature maps from ir feb
            dsired_shape: shape of the rgb fmaps (batch, height, width, channels)
    """
    assert len(fmaps.shape) == 4, "[ERROR] fmaps must have 4 dimensions"
    assert len(desired_shape) == 4, "[ERROR] desired_shape must have 4 dimensions"
    
    desired_height = desired_shape[1]
    desired_width = desired_shape[2]
    pad_height = (desired_height - fmaps.shape[1]) // 2
    pad_width = (desired_width - fmaps.shape[2]) // 2
    paddings = [[0,0], [pad_height, pad_height], [pad_width, pad_width], [0,0]]
    ir_fmaps_padded = tf.pad(fmaps, paddings=paddings, mode="CONSTANT", constant_values=0.0)
    
    return ir_fmaps_padded