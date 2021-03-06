import multiprocessing
import numpy as np
import tensorflow as tf
import os

from tensorflow.python.data import Dataset
from .utils import mkflags


def _to_raw_imgs(img_path, label_path):
    """
    Read/open file and decode image
    """
    return (tf.image.decode_jpeg(tf.read_file(img_path)), 
            tf.image.decode_png(tf.read_file(label_path)))


def setup_train_pipe(config, imgs, labs):
    """
    Build training input data pipe.
    Args:
        config: config dict
        imgs: images/examples
        labs: labels
    """
    FLAGS = mkflags(config)
    cores_count = min(4, max(multiprocessing.cpu_count() // 2, 1))
    n = len(imgs)

    ds_imgs = Dataset.from_tensor_slices(imgs)
    ds_labs = Dataset.from_tensor_slices(labs)
    ds_files = Dataset.zip((ds_imgs, ds_labs))
    # Shuffle all files on each initialization
    ds_files = ds_files.shuffle(n)
    
    # Open images/labels and decode
    ds_t = ds_files.map(
            map_func=_to_raw_imgs,
            num_parallel_calls=cores_count) 
    

    # ========================================
    # ===== CROP AND SCALE
    # Transformations will be faster on smaller imgs?
    # Generate cropping windows
    # This could be done at each epoch but I don't have much more time...
    
    # === TRAINING PIPE RANDOM CROPS & SCALING
    CENTRAL_CROPS = FLAGS.CENTRAL_CROPS 
    RANDOM_CROPS = FLAGS.RANDOM_CROPS 
    T_CROPS_N = RANDOM_CROPS + CENTRAL_CROPS

    CENTRAL_CROPS_LIST = np.linspace(0.5, 1, CENTRAL_CROPS)
    _list = [[0.5 - i/2, 0.5 - i/2, 0.5 + i/2, 0.5 + i/2] for i in CENTRAL_CROPS_LIST]
    _central_boxes = np.array(_list)

    # Random windows will be generated on each training initialization
    y1 = np.random.uniform(0, 0.45, RANDOM_CROPS)
    y2 = np.random.uniform(0.55, 1, RANDOM_CROPS)
    x1 = np.random.uniform(0, 0.45, RANDOM_CROPS)
    x2 = np.random.uniform(0.55, 1, RANDOM_CROPS)
    _rand_boxes = np.array([y1, x1, y2, x2]).transpose()
    t_boxes = np.concatenate((_rand_boxes, _central_boxes), axis=0)
    # Apply all croppings to each single element
    t_boxes_ind = tf.constant([0] * T_CROPS_N)

    # Set network input size
    in_size = tf.constant([FLAGS.INPUT_SZ] * 2)

    # Set batch size to 1, because images has different shapes
    # and we want to generate many images from this one image
    ds_t = ds_t.batch(1)

    ds_t = ds_t.interleave(
            map_func=lambda x, y: Dataset.from_tensors((
                tf.image.crop_and_resize(x, t_boxes, t_boxes_ind, in_size),
                tf.image.crop_and_resize(y, t_boxes, t_boxes_ind, in_size))),
            cycle_length=T_CROPS_N,
            block_length=1).apply(tf.contrib.data.unbatch())
    
    # ===== IMAGE TRANSFORMATIONS 
    trans_num = 2
    def _transform(img, lab):
        fimg = tf.image.flip_left_right(img)
        flab = tf.image.flip_left_right(lab)
        fds = Dataset.from_tensors((fimg, flab))
        ds = Dataset.from_tensors((img, lab))
        return ds.concatenate(fds)

    # === TRAINING PIPE 
    ds_t = ds_t.batch(1)
    ds_t = ds_t.apply(
        tf.contrib.data.parallel_interleave(
            map_func=_transform,
            cycle_length=2 * trans_num,
            block_length=1)).apply(tf.contrib.data.unbatch())


    # ===== SHUFFLING, CONVERSION AND BATCHING
    # First shuffle - acts on unbatched data(!)
    ds_t = ds_t.shuffle(FLAGS.BATCH_SZ ** 2 * T_CROPS_N * trans_num)
    ds_t = ds_t.map(lambda x, y: (x, tf.to_int32(y)))


    # TODO: JŁ 19.07.2018 
    # Check batch/prefetch order correctness?
    # Probably prefetch should be **before** batch
    
    # Create final batches
    ds_t = ds_t.batch(FLAGS.BATCH_SZ)
    ds_t = ds_t.prefetch(buffer_size=FLAGS.BATCH_SZ * 2)

    it_t =  ds_t.make_initializable_iterator()
    return it_t.initializer, it_t.get_next()
