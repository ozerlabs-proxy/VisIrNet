import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

class Warper():
    def __init__(self,
                    batch_size,
                    height_template,
                    width_template):
        self.batch_size=batch_size
        
        self.height_template=height_template
        self.width_template=width_template
        
        self.coords=self.meshgrid(self.batch_size,
                                    self.height_template,
                                    self.width_template)


    def form_unity_matrix(self,matrix_size=3):
        unity_matrix=tf.eye(matrix_size)
        unity_matrix=tf.expand_dims(unity_matrix,axis=0)
        unity_matrix=tf.tile(unity_matrix,[self.batch_size,1,1])
        return unity_matrix
        

    def meshgrid(self, batch, height, width):
        """Construct a 2D meshgrid.

        Args:
            batch: batch size
            height: height of the grid
            width: width of the grid
            matrix: warp matrix explained in projective_inverse_warp
        Returns:
            x,y grid coordinates [batch, 2 , height, width]
        """
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))
        
        x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)

        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)

        return coords

    def meshgrid_after(self,coords, matrix):
        coords=tf.tensordot(matrix,coords,axes = 1)
        coords=coords/(coords[:,2:,:,:]+np.finfo(float).eps)
        coords=coords[:,:2,:,:]

        coords=tf.transpose(coords,[0,2,3,1])
        return coords

    def bilinear_sampler(self,imgs, coords):
        """Construct a new image by bilinear sampling from the input image.

        Points falling outside the source image boundary have value 0.

        Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
            width_t, 2]. height_t/width_t correspond to the dimensions of the output
            image (don't need to be the same as height_s/width_s). The two channels
            correspond to x and y coordinates respectively.
        Returns:
        A new sampled image [batch, height_t, width_t, channels]
        """
        def _repeat(x, n_repeats):
            rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])), 1), [1, 0])
            rep = tf.cast(rep, 'float32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

        with tf.name_scope('image_sampling'):
            coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
            inp_size = imgs.get_shape()
            coord_size = coords.get_shape()
            out_size = coords.get_shape().as_list()
            out_size[3] = imgs.get_shape().as_list()[3]

            coords_x = tf.cast(coords_x, 'float32')
            coords_y = tf.cast(coords_y, 'float32')

            x0 = tf.floor(coords_x)
            x1 = x0 + 1
            y0 = tf.floor(coords_y)
            y1 = y0 + 1

            y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
            x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
            zero = tf.zeros([1], dtype='float32')

            x0_safe = tf.clip_by_value(x0, zero, x_max)
            y0_safe = tf.clip_by_value(y0, zero, y_max)
            x1_safe = tf.clip_by_value(x1, zero, x_max)
            y1_safe = tf.clip_by_value(y1, zero, y_max)

            wt_x0 = x1_safe - coords_x
            wt_x1 = coords_x - x0_safe
            wt_y0 = y1_safe - coords_y
            wt_y1 = coords_y - y0_safe

            ## indices in the flat image to sample from
            dim2 = tf.cast(inp_size[2], 'float32')
            dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
            base = tf.reshape(_repeat( tf.cast(tf.range(coord_size[0]), 'float32') * dim1, coord_size[1] * coord_size[2]),[out_size[0], out_size[1], out_size[2], 1])

            base_y0 = base + y0_safe * dim2
            base_y1 = base + y1_safe * dim2
            idx00 = tf.reshape(x0_safe + base_y0, [-1])
            idx01 = x0_safe + base_y1
            idx10 = x1_safe + base_y0
            idx11 = x1_safe + base_y1

            ## sample from imgs
            imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
            imgs_flat = tf.cast(imgs_flat, 'float32')
            im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
            im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
            im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
            im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

            w00 = wt_x0 * wt_y0
            w01 = wt_x0 * wt_y1
            w10 = wt_x1 * wt_y0
            w11 = wt_x1 * wt_y1

            output = tf.add_n([w00 * im00, w01 * im01, w10 * im10, w11 * im11])
            return output

    def projective_inverse_warp(self,input_feature, matrix):
        pixel_coords = self.meshgrid_after(self.coords, matrix)
        output_img = self.bilinear_sampler(input_feature, pixel_coords )
        return output_img