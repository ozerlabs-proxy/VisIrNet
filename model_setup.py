import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model
import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path



from Tools import BasicModelTools
from Tools import ResnetTools


# ##
def getFeatureEmbeddingBackBone(rgb_inputs_shape,
                                ir_inputs_shape,
                                output_channels_per_block=64,
                                blocks_count =1):
        """
        Feature Embedding Backbone
            return model based on which backbone is requested
        """
        
        # model = BasicModelTools.getFeatureEmbeddingBackBone_Basic(rgb_inputs_shape,
        #                                                         ir_inputs_shape,
        #                                                         output_channels_per_block,
        #                                                         blocks_count)
        model = ResnetTools.getFeatureEmbeddingBackBone_Resnet(rgb_inputs_shape,
                                                                ir_inputs_shape,
                                                                output_channels_per_block)
        

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
                    padding="SAME")(inputs)
    x = layers.Activation("relu")(x)
    
    # x = ResnetTools.resnet_block(x, filters=64, reps =1, strides=1)
    x = ResnetTools.resnet_block(x, filters=128, reps =1, strides=1)
    x = tf.nn.max_pool(x, ksize=2, strides=2, padding='SAME')
    x = ResnetTools.resnet_block(x, filters=256, reps =1, strides=1)
    # x = ResnetTools.conv_batchnorm_relu(x, filters=128, kernel_size=3, strides=1)
    x = tf.nn.max_pool(x, ksize=2, strides=2, padding='SAME')
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME')(x)
    x = layers.BatchNormalization()(x)
    x = ReLU()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = ReLU()(x)
    x = layers.Dense(1024)(x)
    x = layers.Dropout(.2)(x)
    output = layers.Dense(output_size)(x)

    model = keras.Model(inputs=inputs,
                        outputs=output,
                        name="regressionBlock"
                        )

    return model




