import multiprocessing
import numpy as np
import tensorflow as tf
import os

from tensorflow.python.data import Dataset
from .utils import mkflags

def _get_imgs_and_labs(img_path, label_path):
    """
    Read/open file and decode images.
    Network input size is limited.
    Crop image into smaller parts and store data about
    positions to glue all generated predictions. 
    """
    img = tf.image.decode_jpeg(tf.read_file(img_path)) 
    lab = tf.image.decode_png(tf.read_file(label_path))
    img_sz = tf.shape(img)[:2]
    name = tf.string_split([label_path], delimiter="/").values[-1]
    return (
        img,
        lab,
        name,
        img_sz,
    )

def _crop_and_resize_imgs(img, lab, name, shape, boxes, net_sz):
    l = len(boxes)
    inds = [0] * l
    sz = [net_sz] * 2

    def _to_coord(x, h, w):
        x = tf.to_float(x)
        h = tf.to_float(h)
        w = tf.to_float(w)
        return tf.to_int32([
            x[0] * (h - 1), 
            x[1] * (w - 1), 
            x[2] * (h - 1), 
            x[3] * (w - 1)])

    def _to_crops(x, h, w):
        x = tf.to_float(x)
        h = tf.to_float(h)
        w = tf.to_float(w)
        return tf.to_int32([
            x[0] * (h - 1), 
            x[1] * (w - 1), 
            (x[2] - x[0]) * h, 
            (x[3] - x[1]) * w])
 
    def _crop(lab, x):
        return tf.image.crop_to_bounding_box(lab, x[0], x[1], x[2], x[3])


    imgs = Dataset.from_tensors(tf.image.crop_and_resize(img, boxes, inds, sz))
    imgs = imgs.apply(tf.contrib.data.unbatch())

    boxes = Dataset.from_tensor_slices(boxes)
    crops = boxes.map(map_func=lambda x: _to_crops(x, shape[0][0], shape[0][1]))
    boxes = boxes.map(map_func=lambda x: _to_coord(x, shape[0][0], shape[0][1]))

    labs = crops.map(lambda x: _crop(lab, x))
    labs = labs.apply(tf.contrib.data.unbatch())
    
    names = Dataset.from_tensors(name).repeat(l)
    shapes = Dataset.from_tensors(shape[0]).repeat(l)
    return  Dataset.zip((imgs, labs, names, shapes, boxes))


def setup_valid_pipe(config, imgs, labs):
    """
    Build training input data pipe.
    Args:
        config: config dict
        imgs: images/examples
        labs: labels
    """
    FLAGS = mkflags(config)
    VFLAGS = mkflags(FLAGS.VALIDATION)

    # === VALIDATION PIPE DETERMINISTIC CROPS & SCALING
    # Generate cropping boxes which will be used to cover image
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
    ds_labs = Dataset.from_tensor_slices(labs)

    ds_files = Dataset.zip((ds_imgs, ds_labs))
    
    # Open images/labels and decode
    # NOTE: WATCH OUT when using zips along with maps or interleaves...
    ds_ex = ds_files.map(
            map_func=_get_imgs_and_labs,
            num_parallel_calls=cores_count) 

    _crop_rescale = lambda img, lab, name, sh: _crop_and_resize_imgs(img, lab, name, sh, boxes, FLAGS.INPUT_SZ)
    ds_ex = ds_ex.batch(1)
    ds_ex = ds_ex.flat_map(_crop_rescale)


    it =  ds_ex.make_initializable_iterator()
    return it.initializer, it.get_next()

