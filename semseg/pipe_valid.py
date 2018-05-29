import multiprocessing
import numpy as np
import tensorflow as tf
import os

from tensorflow.python.data import Dataset
from .utils import gen_crop_wnds

def _to_raw_imgs(img_path, label_path, IN_SZ):
    """
    Read/open file and decode images.
    Network input size is limited.
    Crop image into smaller parts and store data about
    positions to glue all generated predictions. 
    """
    img = tf.image.decode_jpeg(tf.read_file(img_path)) 
    lab = tf.image.decode_png(tf.read_file(label_path))
    img_sz = tf.shape(img)[:2]
    # Generate half overlapping crops, predictions will be averaged latter 
    crops = tf.py_func(gen_crop_wnds, [img_sz, IN_SZ], tf.int32)
    name = tf.string_split([label_path], delimiter="/").values[-1]
    return (
        img,
        lab,
        name,
        img_sz,
        crops,
        tf.shape(crops)
    )

def setup_valid_pipe(config, imgs, labs):
    """
    Build training input data pipe.
    Args:
        config: config dict
        imgs: images/examples
        labs: labels
    """
    ds_imgs = Dataset.from_tensor_slices(imgs)
    ds_labs = Dataset.from_tensor_slices(labs)

    ds_files = Dataset.zip((ds_imgs, ds_labs))
    
    map_func = lambda x, y: _to_raw_imgs(x, y, config.INPUT_SZ)

    # Open images/labels and decode
    # NOTE: WATCH OUT when using zips along with maps or interleaves...
    ds_v = ds_files.map(
            map_func=map_func,
            num_parallel_calls=cores_count) 

    # Set batch size to 1, because images has different shapes
    # and we want to generate many images from this one image
    
    def _get_crops(img, lab, name, shape, crops, crops_shape):
        ds_crops = Dataset.from_tensor_slices(crops)
        ds_crops = ds_crops.map(lambda x: 
                (tf.image.crop_to_bounding_box(image=img, offset_height=x[0],
                        offset_width=x[1], target_height=x[2],target_width=x[3]),
                tf.image.crop_to_bounding_box(image=lab, offset_height=x[0],
                        offset_width=x[1], target_height=x[2],target_width=x[3]),
                name,
                shape,
                x
                ))
        return ds_crops

    ds_v = ds_v.flat_map(_get_crops)

    it_v =  ds_v.make_initializable_iterator()
    return it_v

