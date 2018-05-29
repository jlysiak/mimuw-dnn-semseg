import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

from .utils import mkflags

def _get_imgs(img_path):
    """
    Read/open file and decode images.
    Network input size is limited.
    Crop image into smaller parts and store data about
    positions to glue all generated predictions. 
    """
    img = tf.image.decode_jpeg(tf.read_file(img_path)) 
    img_sz = tf.shape(img)[:2]
    name = tf.string_split([img_path], delimiter="/").values[-1]
    return (
        img,
        name,
        img_sz,
    )


def _crop_and_resize_imgs(img, name, shape, boxes, net_sz):
    l = len(boxes)
    inds = [0] * l
    sz = [net_sz] * 2

    imgs = Dataset.from_tensors(tf.image.crop_and_resize(img, boxes, inds, sz))
    imgs = imgs.apply(tf.contrib.data.unbatch())
    
    def _to_coord(x, h, w):
        x = tf.to_float(x)
        h = tf.to_float(h)
        w = tf.to_float(w)
        return tf.to_int32([
            x[0] * (h - 1), 
            x[1] * (w - 1), 
            x[2] * (h - 1), 
            x[3] * (w - 1)])

    boxes = Dataset.from_tensor_slices(boxes)
    boxes = boxes.map(map_func=lambda x: _to_coord(x, shape[0][0], shape[0][1]))
    names = Dataset.from_tensors(name).repeat(l)
    shapes = Dataset.from_tensors(shape[0]).repeat(l)
    return  Dataset.zip((imgs, names, shapes, boxes))


def setup_pred_pipe(config, imgs):
    """
    Build input pipe for generating predictions.
    Args:
        config: configuration dict
        imgs: images to predict
    """
    FLAGS = mkflags(config)
    VFLAGS = mkflags(FLAGS.VALIDATION)

    # TODO this is brulat copy-paste, sorry...
    CENTRAL_CROPS = VFLAGS.CENTRAL_CROPS
    AUX_CROPS = VFLAGS.AUX_CROPS
    _aux_crops_t = np.array(AUX_CROPS).transpose()
    _aux_boxes = np.array([
        _aux_crops_t[1], 
        _aux_crops_t[0], 
        _aux_crops_t[3], 
        _aux_crops_t[2]
    ]).transpose()

    V_CROPS_N = len(AUX_CROPS) + CENTRAL_CROPS
    CENTRAL_CROPS_LIST = np.linspace(0.5, 1, CENTRAL_CROPS)
    # Central crops guarantee that whole image will be covered 
    _list = [[0.5 - i/2, 0.5 - i/2, 0.5 + i/2, 0.5 + i/2] for i in CENTRAL_CROPS_LIST]
    _central_boxes = np.array(_list)
    boxes = np.concatenate((_aux_boxes, _central_boxes), axis=0)

    # ==========
    cores_count = min(4, max(multiprocessing.cpu_count() // 2, 1))

    ds_imgs = Dataset.from_tensor_slices(imgs)
    
    # Open images/labels and decode
    # NOTE: WATCH OUT when using zips along with maps or interleaves...
    ds_ex = ds_imgs.map(
            map_func=_get_imgs,
            num_parallel_calls=cores_count) 

    _crop_rescale = lambda img, name, sh: _crop_and_resize_imgs(img, name, sh, boxes, FLAGS.INPUT_SZ)
    ds_ex = ds_ex.batch(1)
    ds_ex = ds_ex.flat_map(_crop_rescale)

    it =  ds_ex.make_initializable_iterator()
    return it.initializer, it.get_next()




    FLAGS = mkflags(config)
    cores_count = min(4, max(multiprocessing.cpu_count() // 2, 1))

    ds_imgs = Dataset.from_tensor_slices(imgs)
     
    _get_imgs_and_crops = lambda x: _read_imgs_gen_crops(x, FLAGS.PREDICTION_SZ)

    # Open images and decode
    # NOTE: WATCH OUT when using zips along with maps or interleaves...
    ds_imgs = ds_imgs.map(
            map_func=_get_imgs_and_crops,
            num_parallel_calls=cores_count) 

    _crop_rescale = lambda img, name, sh, crops: _crop_and_resize_imgs(img, name, sh, crops, FLAGS.INPUT_SZ)
    ds_imgs = ds_imgs.flat_map(_crop_rescale)

    it =  ds_imgs.make_initializable_iterator()
    return it.initializer, it.get_next()



    
