import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

import utils

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
    crops = tf.py_func(utils.gen_crop_wnds, [img_sz, IN_SZ], tf.int32)
    name = tf.string_split([label_path], delimiter="/").values[-1]
    return (
        img,
        lab,
        name,
        img_sz,
        crops,
        tf.shape(crops)
    )

def setup_valid_pipe(config, log):
    imgs = os.listdir(config.DATASET_IMAGES)
    labs = os.listdir(config.DATASET_LABELS)

    n = int(len(imgs) * config.DS_FRAC)
    if n < 2:
        raise Exception("Not enough files in dataset...")
    n_v = int(n * config.VALID_DS_FRAC)
    if n == n_v:
        n_v -= 1
    n_t = n - n_v
    log("Examples in total: %d, train: %d, valid: %d" % (n, n_t, n_v))
    
    # Prevent non-deterministic file listing
    imgs.sort()
    labs.sort()
    imgs = [os.path.join(config.DATASET_IMAGES, el) for el in imgs[:n]]
    labs = [os.path.join(config.DATASET_LABELS, el) for el in labs[:n]]
   
    cores_count = min(4, max(multiprocessing.cpu_count() // 2, 1))
    log("Using %d cores in parallel ops..." % cores_count)

    ds_imgs = Dataset.from_tensor_slices(imgs)
    ds_labs = Dataset.from_tensor_slices(labs)

    ds_files = Dataset.zip((ds_imgs, ds_labs))
    # Shuffle all files
    ds_files = ds_files.shuffle(n)
    ds_files = ds_files.skip(n_t)
    
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

