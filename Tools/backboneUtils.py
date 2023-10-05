import numpy as np
import tensorflow as tf
import cv2



def get_fused_fmaps(feature_maps):
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
    return fineFmaps

def get_fmaps_in_suitable_shape(feature_maps):
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
    return transformed_fmaps
