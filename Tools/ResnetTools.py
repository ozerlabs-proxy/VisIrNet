
"""
Tools necessary to build a customized resnet model


"""
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model



def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def identity_block(tensor, filters):
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = Add()([tensor,x]) # skip connection
    x = ReLU()(x)
    return x

def projection_block(tensor, filters, strides):
         
     #left stream     
     x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)     
     x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)     
     x = Conv2D(filters=4*filters, kernel_size=1, strides=1, padding="same")(x)     
     x = BatchNormalization()(x) 
         
     #right stream     
     shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides, padding="same")(tensor)     
     shortcut = BatchNormalization()(shortcut)          
     x = Add()([shortcut,x])    #skip connection     
     x = ReLU()(x)          
     return x 


def resnet_block(x, filters, reps, strides):
    
    x = projection_block(x, filters, strides)
    for _ in range(reps-1):
        x = identity_block(x,filters)
    return x 


def FeatureEmbeddingBlock_Resnet(inputs, output_channels = 3):

  """
    Get a feature embedding channel given an input shape and output channels
  """
  x = conv_batchnorm_relu(inputs, filters=32, kernel_size=7, strides=1)
  x = MaxPool2D(pool_size = 3, strides =1, padding = "SAME")(x)

  # x = resnet_block(x, filters=32, reps =1, strides=1)
  # x = resnet_block(x, filters=64, reps =2, strides=1)
  # x = resnet_block(x, filters=32, reps =1, strides=1)

  x = resnet_block(x, filters=16, reps =1, strides=1)
  x = resnet_block(x, filters=32, reps =2, strides=1)
  x = resnet_block(x, filters=16, reps =1, strides=1)

  x = conv_batchnorm_relu(x, filters=16, kernel_size=7, strides=1)
  x = Conv2D(filters=output_channels, kernel_size=3, strides=1, padding='SAME')(x)

  output = ReLU()(x)

  return output

def getFeatureEmbeddingBackBone_Resnet(rgb_inputs_shape, ir_inputs_shape, output_channels_per_block=3):
        """
        Feature Embedding Backbone
        """
        # rgb_inputs = Input(shape=rgb_inputs_shape)
        # ir_inputs = Input(shape=ir_inputs_shape)
        
        rgb_inputs = Input(shape=rgb_inputs_shape)
        ir_inputs = Input(shape=ir_inputs_shape)

        rgb_feature_embeddings = FeatureEmbeddingBlock_Resnet(inputs = rgb_inputs,
                                                              output_channels = output_channels_per_block)
        ir_feature_embeddings = FeatureEmbeddingBlock_Resnet(inputs = ir_inputs,
                                                             output_channels = output_channels_per_block)

        model = Model(inputs=(rgb_inputs,ir_inputs), outputs=(rgb_feature_embeddings, ir_feature_embeddings),name="featureEmbeddingBackbone")
        return model
    
"""

  # input = Input(shape=INPUT_SHAPE)
  # x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=2)
  # x = MaxPool2D(pool_size = 3, strides =2)(x)
  # x = resnet_block(x, filters=64, reps =3, strides=1)
  # x = resnet_block(x, filters=128, reps =4, strides=2)
  # x = resnet_block(x, filters=256, reps =6, strides=2)
  # x = resnet_block(x, filters=512, reps =3, strides=2)
  # x = GlobalAvgPool2D()(x)
  # output = Dense(NUMBER_OF_CLASSESS, activation ='softmax')(x)
  # model = Model(inputs=input, outputs=output)

"""