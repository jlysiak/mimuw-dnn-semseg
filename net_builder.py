"""
Network builder
"""

import tensorflow as tf

import net_utils


def _apply_relu(x):
    return tf.nn.relu(x)


def _apply_bnorm(x, t_ind, dfmt):
    """
    Args:
        x: signal
        t_ind: train indicator
        dfmt: data format
    """
    return net_utils.bnorm(x, is_training=t_ind, dfmt=dfmt)


def _apply_conv(x, conf, dfmt):
    """
    Args:
        x: signal
        conf: configuration
            0 - output features
            1 - kernel space dimensions
    """
    l = len(conf)
    f_out = net_utils.get_features_num(x, dfmt)
    spc_dim = 3
    if l > 0:
        f_out = conf[0]
    if l > 1:
        spc_dim = conf[1]
    return net_utils.conv2d(x, features_out=f_out, 
            space_dim=spc_dim, dfmt='NHWC')


def _apply_upconv(x, conf, dfmt, batch_size):
    """
    Args:
        x: signal
        conf: config
    """
    l = len(conf)
    f_out = net_utils.get_features_num(x, dfmt)
    spc_dim = 3
    if l > 0:
        f_out = conf[0]
    if l > 1:
        spc_dim = conf[1]

    return net_utils.upconv2d(x, 
            features_out=f_out, 
            dfmt=dfmt, 
            space_dim=spc_dim, 
            batch_size=batch_size)


def _apply_resize(x, new_size, dfmt):
    apply_trans = True if dfmt == 'NCHW' else False
    if apply_trans: 
        x = tf.transpose(x, perm=[0, 3, 1, 2])
    x = tf.image.resize_nearest_neighbor(x,
                size=[new_size] * 2,
                align_corners=True)
    if apply_trans: 
        x = tf.transpose(x, perm=[0, 2, 3, 1])
    return x


def _apply_concat(x1, ts, conf, dfmt):
    if len(conf) > 0:
        ind = conf[0]
        x2 = ts[ind]
    else:
        raise Exception("Tensor concat requires one argument!")

    axis = 3
    if dfmt == 'NCHW':
        axis = 1
    return tf.concat([x1, x2], axis=axis)


def _apply_pool(x, conf, dfmt):
    ksize = 2
    if len(conf) > 0:
        ksize = conf[0]
    return net_utils.max_pool(x, ksize=ksize, dfmt=dfmt)


def build_network(x, batch_size, arch_list, log,train_indicator=True, dfmt='NHWC'):
    """ 
    NETWORK BUILDER 
    Args:
        x: input in `NWHC` format
        arch_list: architecture description
        train_indicator: training phase indicator
    """
    ch_first = False if dfmt == 'NHWC' else True

    if ch_first: 
        _x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = tf.reshape(_x, shape=[-1, 3, 256, 256])
    else:
        x = tf.reshape(x, shape=[-1, 256, 256, 3])

    i = 0
    signal = x     # Main data path
    layers = [signal]   # Layers stack
    # Annotated layers map
    layers_annotated = dict()
    
    for el in arch_list:
        args = el[0].split(":")
        ltype = args[0]

        if ltype == 'relu':
            signal = _apply_relu(signal)
        elif ltype == 'bnorm':
            signal = _apply_bnorm(signal, 
                        t_ind=train_indicator,
                        dfmt=dfmt)
        elif ltype == 'conv':
            signal = _apply_conv(signal, 
                        conf=el[1:],
                        dfmt=dfmt)
        elif ltype == 'upconv':
            signal = _apply_upconv(signal,
                        conf=el[1:],
                        dfmt=dfmt,
                        batch_size=batch_size)
        elif ltype == 'concat':
            signal = _apply_concat(signal, 
                        ts=layers_annotated, 
                        conf=el[1:], 
                        dfmt=dfmt)
        elif ltype == 'pool':
            signal = _apply_pool(signal, 
                        conf=el[1:],
                        dfmt=dfmt)
        elif ltype == 'resize':
            signal = _apply_resize(signal, 
                        conf=el[1:],
                        dfmt=dfmt)
        else:
            raise Exception("Layer type not recognized!")

        if len(args) > 1:
            _id = int(args[1])
            layers_annotated[_id] = signal

        i += 1
        layers += [signal]
        log("Created: " + str(signal))    
    
    if ch_first: 
        signal = tf.transpose(signal, perm=[0, 2, 3, 1])
    
    return signal
