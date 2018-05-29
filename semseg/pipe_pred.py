import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

from .utils import mkflags, gen_crop_wnds

def _read_imgs_gen_crops(img_path, in_sz):
    """
    Read/open file and decode images.
    Network input size is limited.
    Crop image into smaller parts and store data about
    positions to glue all generated predictions. 
    """
    img = tf.image.decode_jpeg(tf.read_file(img_path)) 
    img_sz = tf.shape(img)[:2]
    crops = tf.py_func(gen_crop_wnds, [img_sz, in_sz], tf.int32)
    name = tf.string_split([img_path], delimiter="/").values[-1]
    return (
        img,
        name,
        img_sz,
        crops
    )


def _crop_and_resize_imgs(img, name, shape, crops, net_input_sz):
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
    ds_crops = ds_crops.map(lambda img, _name, _sh, _wnd: 
            (tf.image.resize_images(images=img, size=[net_input_sz] * 2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
            _name,
            _sh,
            _wnd))
    return ds_crops



def setup_pred_pipe(config, imgs):
    """
    Build input pipe for generating predictions.
    Args:
        config: configuration dict
        imgs: images to predict
    """
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



    
