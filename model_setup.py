"""
Will include functions to build the models at different levels of abstraction.
"""

# ##
import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

# ##
"""residual_block"""
def residual_block(x, 
                    number_of_filters, 
                    match_filter_size=False):
    """
    Residual block for ResNet
    """
    x_skip = x

    x = layers.Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1,1), padding="same")(x_skip)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(number_of_filters, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)

    if match_filter_size:
        x_skip = layers.Lambda(lambda x: tf.pad(x[:,:,:,:],
                                            tf.constant([[0, 0,], 
                                                            [0, 0], 
                                                            [0, 0], 
                                                            [number_of_filters//4, number_of_filters//4]]),
                                            mode="CONSTANT"))(x_skip)
    x = layers.Add()([x, x_skip])
    # Nonlinearly activate the result
    x = layers.Activation("relu")(x)
    # Return the result
    return x

""" residual_blocks stack """
def residual_blocks(x,
                    _filters_count=64, 
                    blocks_count=3):
    """
    stack of residual blocks for ResNet
    """
    for layer_group in range(blocks_count):
        for block in range(2):
            if layer_group > 0 and block == 0:
                _filters_count *= 2
                x = residual_block(x, _filters_count, match_filter_size=True)
            else:
                x = residual_block(x, _filters_count)
    return x
# ## 
def FeatureEmbeddingBlock(inputs,
                            output_channels = 64,
                            blocks_count=1):
    """
    Feature Embedding Block 
    """

    # inputs = layers.Input(shape=input_shape)
    initial_feature_maps = 64
    x = layers.Conv2D(initial_feature_maps, 
                        kernel_size=(3,3),
                        strides=(1,1), 
                        padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_blocks(x=x,
                        _filters_count=initial_feature_maps,
                        blocks_count=blocks_count
                        )

    outputs = layers.Conv2D(output_channels,
                            kernel_size=(3,3),
                            strides=(1,1), 
                            padding="same")(x)

    return outputs

# ##
def getFeatureEmbeddingBackBone(rgb_inputs_shape,
                                ir_inputs_shape,
                                output_channels_per_block=64):
        """
        Feature Embedding Backbone
        """
        rgb_inputs = layers.Input(shape=rgb_inputs_shape)
        ir_inputs = layers.Input(shape=ir_inputs_shape)

        rgb_feature_embeddings = FeatureEmbeddingBlock(inputs = rgb_inputs, 
                                                        output_channels = output_channels_per_block)
        # rgb_feature_embedding_block = keras.Model(inputs=rgb_inputs,
        #                                   outputs=rgb_feature_embeddings,
        #                                   name="rgb_feature_embedding_block"
        #                                   )


        ir_feature_embeddings = FeatureEmbeddingBlock(inputs = ir_inputs, 
                                                        output_channels = output_channels_per_block)
        # ir_feature_embedding_block= keras.Model(inputs=ir_inputs,
        #                                   outputs=ir_feature_embeddings,
        #                                   name="ir_feature_embedding_block"
        #                                   )


        model = keras.Model(inputs=(rgb_inputs,ir_inputs),
                        outputs=(rgb_feature_embeddings, ir_feature_embeddings),
                        name="featureEmbeddingBackbone"
                        )
        return model

# ##
## regression block

def getRegressionHead(input_shape,
                        output_size=8,
                        blocks_count =1
                        ):
    """
    Regression Head

    """

    inputs = layers.Input(shape=input_shape)

    initial_feature_maps = 64
    x = layers.Conv2D(initial_feature_maps, 
                    kernel_size=(3,3),
                    strides=(1,1), 
                    padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_blocks(x=x,
                        _filters_count=initial_feature_maps,
                        blocks_count=blocks_count)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Dropout(.2)(x)
    x = layers.Dense(output_size)(x)

    model = keras.Model(inputs=inputs,
                        outputs=x,
                        name="regressionBlock"
                        )

    return model




