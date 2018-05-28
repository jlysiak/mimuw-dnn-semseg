import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

import utils

def _read_imgs_gen_crops(img_path, in_sz):
    """
    Read/open file and decode images.
    Network input size is limited.
    Crop image into smaller parts and store data about
    positions to glue all generated predictions. 
    """
    img = tf.image.decode_jpeg(tf.read_file(img_path)) 
    img_sz = tf.shape(img)[:2]
    crops = tf.py_func(_conv, [img_sz, in_sz], tf.int32)
    name = tf.string_split([label_path], delimiter="/").values[-1]
    return (
        img,
        name,
        img_sz,
        crops
    )


def _crop_imgs(img, name, shape, crops):
    """
    Crop given image using crops boxes
    Returns:
        dataset of tuples
        (cropped image, src img name, src img shape, crop params)
    """
    ds_crops = Dataset.from_tensor_slices(crops)
    ds_crops = ds_crops.map(lambda x: 
            (tf.image.crop_to_bounding_box(image=img, offset_height=x[0],
                    offset_width=x[1], target_height=x[2],target_width=x[3]),
            name,
            shape,
            x))
    return ds_crops



def setup_pred_pipe(imgs_paths, config, log):
    """
    Build input pipe for generating predictions.
    """
    cores_count = min(4, max(multiprocessing.cpu_count() // 2, 1))
    log("Using %d cores in parallel ops..." % cores_count)

    ds_imgs = Dataset.from_tensor_slices(imgs_paths)
     
    _get_imgs_and_crops = lambda x, y: _read_imgs_gen_crops(x, y, config.INPUT_SZ)

    # Open images and decode
    # NOTE: WATCH OUT when using zips along with maps or interleaves...
    ds_imgs = ds_imgs.map(
            map_func=_get_imgs_and_crops,
            num_parallel_calls=cores_count) 

    ds_v = ds_v.flat_map(_crop_imgs)

    it_v =  ds_v.make_initializable_iterator()
    return it_v



    
