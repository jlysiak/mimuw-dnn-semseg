import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
import os


class CONF:
    def __init__(self, kw):
        self.__dict__ = kw

def _to_raw_imgs(img_path, label_path):
    """
    Read/open file and decode image
    """
    return (tf.image.decode_jpeg(tf.read_file(img_path)), 
            tf.image.decode_png(tf.read_file(label_path)))


def setup_train_valid_pipes(config, log):
    """
    Creates input pipe.
    Args:
        config: trainer configuration
        frac: fraction of all images to take
    Returns:
        Two iterators for train and validation dataset
    """
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
    
    # Open images/labels and decode
    ds_examples = ds_files.map(
            map_func=_to_raw_imgs,
            num_parallel_calls=cores_count) 
    
    # Split into train and validation datasets
    ds_t = ds_examples.take(n_t)
    ds_v = ds_examples.skip(n_t)

    # ========================================
    # ===== CROP AND SCALE
    # Transformations will be faster on smaller imgs?
    # Generate cropping windows
    # This could be done at each epoch but I don't have much more time...
    
    # === TRAINING PIPE RANDOM CROPS & SCALING
    CENTRAL_CROPS = config.CENTRAL_CROPS 
    RANDOM_CROPS = config.RANDOM_CROPS 
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
    t_boxes_ind = tf.constant([0 for _ in range(T_CROPS_N)])


    # === VALIDATION PIPE DETERMINISTIC CROPS & SCALING
    vconfig = CONF(config.VALIDATION)
    CENTRAL_CROPS = vconfig.CENTRAL_CROPS
    AUX_CROPS = vconfig.AUX_CROPS
    _aux_crops_t = np.array(AUX_CROPS).transpose()
    _aux_boxes = np.array([
        _aux_crops_t[1], 
        _aux_crops_t[0], 
        _aux_crops_t[3], 
        _aux_crops_t[2]
    ]).transpose()

    V_CROPS_N = len(AUX_CROPS) + CENTRAL_CROPS
    CENTRAL_CROPS_LIST = np.linspace(0.5, 1, CENTRAL_CROPS)
    _list = [[0.5 - i/2, 0.5 - i/2, 0.5 + i/2, 0.5 + i/2] for i in CENTRAL_CROPS_LIST]
    _central_boxes = np.array(_list)
    
    v_boxes = np.concatenate((_aux_boxes, _central_boxes), axis=0)
    v_boxes_ind = tf.constant([0 for _ in range(V_CROPS_N)])
    # ==========

    # Set network input size
    in_size = tf.constant([config.INPUT_SZ] * 2)
    out_size = tf.constant([config.OUTPUT_SZ] * 2)

    # Set batch size to 1, because images has different shapes
    # and we want to generate many images from this one image
    ds_t = ds_t.batch(1)
    ds_v = ds_v.batch(1)

    ds_t = ds_t.interleave(
            map_func=lambda x, y: Dataset.from_tensors((
                tf.image.crop_and_resize(x, t_boxes, t_boxes_ind, in_size),
                tf.image.crop_and_resize(y, t_boxes, t_boxes_ind, out_size))),
            cycle_length=T_CROPS_N,
            block_length=1).apply(tf.contrib.data.unbatch())

    v_trans_num = V_CROPS_N
    ds_v = ds_v.interleave(
            map_func=lambda x, y: Dataset.from_tensors((
                tf.image.crop_and_resize(x, v_boxes, v_boxes_ind, in_size),
                tf.image.crop_and_resize(y, v_boxes, v_boxes_ind, out_size))),
            cycle_length=1,
            block_length=V_CROPS_N).apply(tf.contrib.data.unbatch())
    
    # ========================================
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


    # === VALIDATION PIPE 
    if vconfig.FLIP_LR:
        v_trans_num *= 2
        ds_v = ds_v.batch(1)
        ds_v = ds_v.apply(
                tf.contrib.data.parallel_interleave(
                    map_func=_transform,
                    cycle_length=1,
                    block_length=trans_num)).apply(tf.contrib.data.unbatch())

    # ===== SHUFFLING, CONVERSION AND BATCHING
    # First shuffle - acts on unbatched data(!)
    ds_t = ds_t.shuffle(config.BATCH_SZ ** 2 * T_CROPS_N * trans_num)
    ds_t = ds_t.map(lambda x, y: (x, tf.to_int32(y)))

    ds_v = ds_v.map(lambda x, y: (x, tf.to_int32(y)))

    # Create final batches
    ds_t = ds_t.batch(config.BATCH_SZ)
    ds_t = ds_t.prefetch(buffer_size=config.BATCH_SZ * 2)

    it_t =  ds_t.make_initializable_iterator()
    it_v =  ds_v.make_initializable_iterator()
    return it_t, it_v, v_trans_num
