"""
Network builder simple utils.
"""
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    try:
        stddev = np.prod(shape) ** (-0.5)
    except:
        stddev = np.prod(shape).value ** (-0.5)
    initializer = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initializer)


def bnorm(x, is_training=True, act=None, dfmt='NHWC'): 
    return tf.contrib.layers.batch_norm(x,
            scale=True,
            center=True,
            fused=True,
            is_training=is_training,
            activation_fn=act,
            data_format=dfmt)


def conv2d(x, features_out, space_dim=3, dfmt='NHWC'):
    strides = [1] * 4
    shape = [space_dim] * 2
    features_in = x.shape[3:].as_list() if dfmt == 'NHWC' else x.shape[1:2].as_list()
    shape_features = features_in + [features_out]
    if dfmt == 'NCHW': # GPU + MKL recommended
        shape_features.reverse()

    shape += shape_features
    W = weight_variable(shape)
    return tf.nn.conv2d(x, W, 
            strides=strides,
            padding='SAME',
            data_format=dfmt,
            use_cudnn_on_gpu=True)


def upconv2d(x, features_out, dfmt='NHWC', space_dim=3, batch_size=None):
    if batch_size is None:
        batch_size = x.shape[0].value 
    strides = [1, 2, 2, 1] if dfmt == 'NHWC' else [1, 1, 2, 2]
    shape = [space_dim] * 2

    features_in = x.shape[3:].as_list() if dfmt == 'NHWC' else x.shape[1:2].as_list()
    space_shape = x.shape[1:3].as_list() if dfmt == 'NHWC' else x.shape[2:].as_list()

    features_shape = [features_out] + features_in
    
    _shape = [2 * x for x in space_shape]
    if dfmt == 'NCHW':
        out_shape = [batch_size] + [features_out] + _shape
        shape += [features_out] + features_in
    else: # NHWC
        out_shape = [batch_size] + _shape + [features_out]
        shape += features_in + [features_out] 

    W = weight_variable(shape)
    return tf.nn.conv2d_transpose(x, W,
        output_shape=out_shape,
        strides=strides, 
        padding='SAME',
        data_format=dfmt)


def max_pool(x, ksize=2, dfmt='NHWC'):
    if dfmt == 'NHWC':
        k = [1] + [ksize] * 2 + [1]
    else:
        k = [1, 1] + [ksize] * 2
    return tf.nn.max_pool(x, 
        ksize=k, 
        strides=k, 
        padding='SAME',
        data_format=dfmt)

def get_features_num(x, dfmt):
    if dfmt == 'NHWC':
        return x.shape.as_list()[3]
    else: # NCWH
        return x.shape.as_list()[1]

