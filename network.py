"""
Deep Neural Networks @ MIMUW 2017/18
Jacek Łysiak

Semantic segmentation with convolutional network

"""
import tensorflow as tf
from tensorflow.python.data import Dataset
import numpy as np
import math
import os
import argparse
import matplotlib.pyplot as plt 
import shutil
import datetime
import sys
import multiprocessing

from PIL import Image

class Trainer(object):
    """
    Main trainer class
    """
    
    DEF_CKPT_DIR    = "checkpoint"
    DEF_CKPT_NAME   = "semseg.ckpt"
    DEF_LOG_PATH    = "/tmp/semseg-train.log"
    DEF_TB_DIR      = "/tmp/semseg-tb-log"

    IMAGES_DIR      = "images"
    LABELS_DIR      = "labels_plain"

    VALID_DS_FRAC   = 0.1
    BATCH_SZ        = 16
    INPUT_SZ        = 256

    def __init__(self, dataset_dir, ckpt_dir=None, log_path=None, tb_dir=None):
        if dataset_dir is None:
            raise Exception("Dataset directory not provided!")

        self.dataset_dir = dataset_dir
        self.dataset_images = os.path.join(dataset_dir, Trainer.IMAGES_DIR)
        self.dataset_labels = os.path.join(dataset_dir, Trainer.LABELS_DIR)
        self.ckpt_dir = Trainer.DEF_CKPT_DIR if ckpt_dir is None else ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.ckpt_path = os.path.join(self.ckpt_dir, Trainer.DEF_CKPT_NAME)
        
        self.log_path = Trainer.DEF_LOG_PATH if log_path is None else log_path
        self.log_file = open(self.log_path, "a")

        self.tb_dir = Trainer.DEF_TB_DIR if tb_dir is None else tb_dir
        self.t_tb_writer = tf.summary.FileWriter(
                os.path.join(self.tb_dir, "train"))
        self.v_tb_writer = tf.summary.FileWriter(
                os.path.join(self.tb_dir, "valid"))
        
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log_file.close()
    
    def log(self, s):
        timestamp =  '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        s = "[{:>19s}] {}\n".format(timestamp, s)
        sys.stderr.write(s)
        self.log_file.write(s)

    def setup_input_pipes(self, **kw):
        """
        Wraper for input pipe creator.
        Places pipe on cpu in proper name scope.
        """

        with tf.device("/cpu:0"):
            with tf.name_scope("input_pipe"):
                return self._setup_input_pipes(**kw)

    def _to_raw_imgs(img_path, label_path):
        """
        Read/open file and decode image
        """
        return (tf.image.decode_jpeg(tf.read_file(img_path)), 
                tf.image.decode_png(tf.read_file(label_path)))


    def _setup_input_pipes(self, frac=1.):
        """
        Creates input pipe.
        Args:
            frac: fraction of all images to take
        Returns:
            Two iterators for train and validation dataset
        """
        imgs = os.listdir(self.dataset_images)
        labs = os.listdir(self.dataset_labels)

        n = int(len(imgs) * frac)
        if n < 2:
            raise Exception("Not enough files in dataset...")
        n_v = int(n * Trainer.VALID_DS_FRAC)
        if n == n_v:
            n_v -= 1
        n_t = n - n_v
        self.log("Examples in total: %d, train: %d, valid: %d" % (n, n_t, n_v))
        
        # Prevent non-deterministic file listing
        imgs.sort()
        labs.sort()
        imgs = [os.path.join(self.dataset_images, el) for el in imgs[:n]]
        labs = [os.path.join(self.dataset_labels, el) for el in labs[:n]]
       
        cores_count = max(multiprocessing.cpu_count() // 2, 1)
        self.log("Using %d cores in input pipe..." % cores_count)

        ds_imgs = Dataset.from_tensor_slices(imgs)
        ds_labs = Dataset.from_tensor_slices(labs)
        ds_files = Dataset.zip((ds_imgs, ds_labs))
        # Shuffle all files
        ds_files = ds_files.shuffle(n)
        
        # Open images/labels and decode
        ds_examples = ds_files.map(
                map_func=Trainer._to_raw_imgs,
                num_parallel_calls=cores_count) 
        
        # Split into train and validation datasets
        ds_t = ds_examples.take(n_t)
        ds_v = ds_examples.skip(n_t)

        # ========================================
        # == Crop and scale
        # == Transformations will be faster on smaller imgs?

        # Generate cropping windows
        # NOTE Get randomly images corners and WH
        # This could be done at each epoch but I don't have much more time...
        # NOTE Here I can do many crops from different places
        
        RANDOM_CROPS = 10
        CENTRAL_CROPS = 5
        CROPS_N = RANDOM_CROPS + CENTRAL_CROPS
        CENTRAL_CROPS_LIST = np.linspace(0.5, 1, CENTRAL_CROPS)
        _list = [[0.5 - i/2, 0.5 - i/2, 0.5 + i/2, 0.5 + i/2] for i in CENTRAL_CROPS_LIST]
        _central_boxes = np.array(_list)
        
        y1 = np.random.uniform(0, 0.45, RANDOM_CROPS)
        y2 = np.random.uniform(0.55, 1, RANDOM_CROPS)
        x1 = np.random.uniform(0, 0.45, RANDOM_CROPS)
        x2 = np.random.uniform(0.55, 1, RANDOM_CROPS)
        _rand_boxes = np.array([y1, x1, y2, x2]).transpose()
        boxes = np.concatenate((_rand_boxes, _central_boxes), axis=0)

        # Apply all croppings to each single element
        boxes_ind = tf.constant([0 for _ in range(CROPS_N)])
        # Set final input size
        out_size = tf.constant([Trainer.INPUT_SZ] * 2)

        # Set batch size to 1, because images has different shapes
        # and we want to generate many images from this one image
        ds_t = ds_t.batch(1)
        ds_v = ds_v.batch(1)

        ds_t = ds_t.interleave(
                map_func=lambda x, y: Dataset.from_tensors((
                    tf.image.crop_and_resize(x, boxes, boxes_ind, out_size),
                    tf.image.crop_and_resize(y, boxes, boxes_ind, out_size))),
                cycle_length=1,
                block_length=CROPS_N).apply(tf.contrib.data.unbatch())

        ds_v = ds_v.interleave(
                map_func=lambda x, y: Dataset.from_tensors((
                    tf.image.crop_and_resize(x, boxes, boxes_ind, out_size),
                    tf.image.crop_and_resize(y, boxes, boxes_ind, out_size))),
                cycle_length=1,
                block_length=CROPS_N).apply(tf.contrib.data.unbatch())
        
        # ========================================
        # == Data set transformations 
    
        ds_t = ds_t.batch(1)
        ds_v = ds_v.batch(1)
        
        trans_num = 2
        def _transform(img, lab):
            fimg = tf.image.flip_left_right(img)
            flab = tf.image.flip_left_right(lab)
            fds = Dataset.from_tensors((fimg, flab))
            ds = Dataset.from_tensors((img, lab))
            return ds.concatenate(fds)

        ds_t = ds_t.apply(
            tf.contrib.data.parallel_interleave(
                map_func=_transform,
                cycle_length=2 * trans_num,
                block_length=1)).apply(tf.contrib.data.unbatch())

        ds_v = ds_v.apply(
                tf.contrib.data.parallel_interleave(
                    map_func=_transform,
                    cycle_length=1,
                    block_length=trans_num)).apply(tf.contrib.data.unbatch())

        ds_t = ds_t.map(lambda x, y: (x, tf.to_int32(y)))
        ds_v = ds_v.map(lambda x, y: (x, tf.to_int32(y)))
        ds_t = ds_t.batch(Trainer.BATCH_SZ)
        # ========================================
        # == Final cropping and scaling
        it_t =  ds_t.make_initializable_iterator()
        it_v =  ds_v.make_initializable_iterator()
        return it_t, it_v

    def test_input_pipe(self, out_dir="pipe_test", train_outs="training", 
            valid_outs="validation", frac=1.):
        """
        Handy method to check input pipe.
        """
        t_outs = os.path.join(out_dir, train_outs)
        v_outs = os.path.join(out_dir, valid_outs)
        if not os.path.exists(t_outs):
            os.makedirs(t_outs)
        if not os.path.exists(v_outs):
            os.makedirs(v_outs)

        self.log("******* Input pipe test")
        it_t, it_v = self.setup_input_pipes(frac=frac)
        next_t = it_t.get_next()
        next_v = it_v.get_next()
         
        with tf.Session() as sess:
            sess.run([it_t.initializer, it_v.initializer])
            try:
                i = 0
                while True:
                    a, b = sess.run(next_t)
                    print(i, a.shape, b.shape)
                    for k in range(len(a)):
                        result = Image.fromarray(a[k].astype(np.uint8))
                        result.save(os.path.join(t_outs, "%d-a.jpg" % i))
                        result = Image.fromarray(b[k].astype(np.uint8).reshape(256,256), mode="L")
                        result.save(os.path.join(t_outs, "%d-b.png" % i))
                        i += 1
            except tf.errors.OutOfRangeError:
                print("End...")

            try:
                i = 0
                while True:
                    a, b = sess.run(next_v)
                    print(i, a.shape, b.shape)
                    #for k in range(len(a)):
                    result = Image.fromarray(a.astype(np.uint8))
                    result.save(os.path.join(v_outs, "%d-a.jpg" % i))
                    result = Image.fromarray(b.astype(np.uint8).reshape(256,256), mode="L")
                    result.save(os.path.join(v_outs, "%d-b.png" % i))
                    i += 1
            except tf.errors.OutOfRangeError:
                print("End...")


    def build_network(self, x_iter, y_iter):
        """
        Build neural network.
        """
        #with tf.device("/gpu:0"):
        with tf.name_scope("network"):
            logits = self._build_network(x_iter)
        pred = tf.argmax(logits, axis=3, output_type=tf.int32)
        
        # Save one image, label and prediction from each batch
        t_summaries = []
        t_summaries += [tf.summary.image("train/imgs/label", 
            tf.cast(y_iter, tf.uint8), max_outputs=1)]
        t_summaries += [tf.summary.image("train/imgs/image", 
            x_iter, max_outputs=1)]
        t_summaries += [tf.summary.image("train/imgs/prediction", 
            tf.cast(tf.reshape(pred, [-1, 256, 256, 1]), tf.uint8), max_outputs=1)]
        
        y_iter = tf.reshape(y_iter, [-1, 256, 256])
        
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_iter), tf.float32))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_iter,
                logits=logits))
        # Add batch loss and accuracy
        t_summaries += [tf.summary.scalar("train/acc", acc)]
        t_summaries += [tf.summary.scalar("train/loss", loss)]
        
        # TRAINING OPS - within batch
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.MomentumOptimizer(0.02, 
                   momentum=0.9).minimize(loss)
         
        
        # VALIDATION OPS - cumulative across dataset

        # We can take mean of means over batches beacuse 
        # all batches has same size.
        v_acc, v_acc_update = tf.metrics.mean(
                values=acc,
                name="valid/metrics/acc")
        v_loss, v_loss_update = tf.metrics.mean(
                values=loss,
                name="valid/metrics/loss")

        # Generate initializer for statistics over validation set
        metrics_vars = []
        for el in tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                scope="valid/metrics"):
            metrics_vars.append(el)
        metrics_init = tf.variables_initializer(var_list=metrics_vars)

        # TRAIN OPS
        self.t_step = train_step
        self.t_loss = loss
        self.t_acc = acc
        self.t_summaries = tf.summary.merge(t_summaries)

        # VALIDATION OPS
        self.v_acc = v_acc
        self.v_loss = v_loss
        self.v_update = [v_acc_update, v_loss_update]
        self.v_init = metrics_init

    def _build_network(self, x_iter):
        """
        Change `NHWC` into `NCHW` for better performance on GPUS.
        """

        # == BUILDER HELPERS
        def bnorm(x, ind, act=None):     
            return tf.contrib.layers.batch_norm(x,
                scale=True,
                center=True,
                fused=True,
                is_training=ind,
                activation_fn=act)

        def weight_variable(shape):
            try:
                stddev = np.prod(shape) ** (-0.5)
            except:
                stddev = np.prod(shape).value ** (-0.5)
            initializer = tf.truncated_normal(shape, stddev=stddev)
            return tf.Variable(initializer, name='weight')

        def conv2d(x, out_sz):
            shape = [3, 3] + x.shape[3:].as_list() + [out_sz] # changed
            W = weight_variable(shape)
            return tf.nn.conv2d(x, W, 
                strides=[1] * 4, 
                padding='SAME',
                data_format='NHWC',  # changed
                use_cudnn_on_gpu=True)
        
        def upconv2d(x, out_sz):
            shape = [3, 3] + [out_sz] + x.shape[3:].as_list() # changed
            out_shape = [Trainer.BATCH_SZ] + [2 * x for x in x.shape[1:3].as_list()] + [out_sz] 
            W = weight_variable(shape)
            return tf.nn.conv2d_transpose(x, W,
                output_shape = out_shape,
                strides=[1, 2, 2, 1], # NOTE 1 1 2 2 in NCHW
                padding='SAME',
                data_format='NHWC') # changed

        def pool(x):
            k = [1, 2, 2, 1] # changed
            return tf.nn.max_pool(x, 
                    ksize=k, 
                    strides=k, 
                    padding='SAME',
                    data_format='NHWC') # changed

        # === NETWORK BUILDER 
        train_indicator = tf.Variable(True,
            trainable=False,
            name="TRAIN_INDICATOR")
        
        net_conf = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        # _x_iter = tf.transpose(x_iter)
        #x_iter = tf.reshape(x_iter, shape=[-1, 3, 256, 256])
        x_iter = tf.reshape(x_iter, shape=[-1, 256, 256, 3])
        signal = x_iter
        layers = [signal]
        i = 0
        
        self.log("******* Layers:")
        self.log("Input:")
        self.log("%d - %s" % (i, str(signal)))

        self.log("** Downsampling layers")
        for n in net_conf:
            i += 1
            self.log("Layer: %d" % i)
            signal = conv2d(signal, n)
            self.log("%s" % str(signal))
            signal = bnorm(signal, train_indicator, tf.nn.relu)
            self.log("%s" % str(signal))
            signal = pool(signal)
            self.log("%s" % str(signal))
            layers.append(signal)

        self.log("** Upsampling layers")
        layers.reverse()
        net_conf.reverse()
        for layer, n in zip(layers[1:], net_conf[1:]):
            i += 1
            self.log("Layer: %d" % i)
            signal = upconv2d(signal, n)
            self.log("%s" % str(signal))
            signal = bnorm(signal, train_indicator, tf.nn.relu)
            self.log("%s" % str(signal))
            signal = tf.concat([signal, layer], axis=3)
            self.log("%s" % str(signal))
            signal = conv2d(signal, n)
            self.log("%s" % str(signal))
            signal = bnorm(signal, train_indicator, tf.nn.relu)
            self.log("%s" % str(signal))
        
        self.log("Output:")
        signal = upconv2d(signal, 64)
        self.log("%s" % str(signal))
        signal = bnorm(signal, train_indicator, tf.nn.relu)
        self.log(signal)
        signal = tf.concat([signal, x_iter], axis=3)
        self.log(signal)
        
        shape = [3, 3] + signal.shape[3:].as_list() + [66] # changed
        W = weight_variable(shape)
        signal = tf.nn.conv2d(signal, W, 
            strides=[1] * 4, 
            padding='SAME',
            data_format='NHWC', # changes
            use_cudnn_on_gpu=True)
        self.log(signal)
        signal = bnorm(signal, train_indicator, tf.nn.relu)
        self.log(signal)
        #signal = tf.transpose(signal, perm=[0, 2, 3, 1])
        self.log(signal)
        self.log("****** END OF BUILDER")
        return signal


    def train(self, files_frac=1.0):
        """
        Train model.
        Args:
            [files_frac]: fraction of files used in training
        """
        self.log("******* Starting training procedure...")
        it_t, it_v = self.setup_input_pipes(frac=files_frac)
        next_t = it_t.get_next()
        next_v = it_v.get_next()
        
        self.build_network(next_t[0], next_t[1])
        
        saver = tf.train.Saver()
        restore = True
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            restore = False

        with tf.Session() as sess:
            self.log("Initialize training dataset...")
            it_t.initializer.run()

            if restore:
                try:
                    self.log("Attempt to restore model from: %s" % self.ckpt_path)
                    saver.restore(sess, self.ckpt_path)
                    self.log("DONE!")
                except:
                    self.log("Cannot restore... Initializing new variables")
                    tf.global_variables_initializer().run()
            else:
                self.log("Initializing new network variables")
                tf.global_variables_initializer().run()

            epochs = 1
            batch_n = 0
            self.log("**** TRAINING STARTED")
            try:
                for epoch in range(epochs):
                    self.log("** Epoch: %d" % (epoch + 1))
                    while True:
                        t_loss, t_acc, t_summ, _ = sess.run([
                            self.t_loss, 
                            self.t_acc, 
                            self.t_summaries, 
                            self.t_step])
                        batch_n += 1
                        self.log("batch: %d, loss: %f, accuracy: %1.3f" % 
                                    (batch_n, t_loss, t_acc))
                        self.t_tb_writer.add_summary(t_summ, batch_n)

                        if batch_n % 100 == 0:
                            self.log("**** VALIDATION")
            except KeyboardInterrupt:
                self.log("Training stopped by keyboard interrupt!")
            
            save_path = saver.save(sess, self.ckpt_path)
            self.log("Model saved as: %s" % save_path)




    def predict(self, outdir):
        pass

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Deep convolutional network\
            for semantic segmentation - Deep Neural Networks @ MIMUW 2017")
    args.add_argument("-d", "--dataset", help="Dataset location", 
            metavar="DIR")
    args.add_argument("-c", "--checkpoint", help="Checkpoint directory")
    args.add_argument("-l", "--logs", help="Log file path")
    args.add_argument("-b", "--tblogs", help="Directory for TensorBoard logs")
    args.add_argument("-r", "--lrate", help="Learning rate")
    args.add_argument("-f", "--frac", help="Fraction of dataset taken.", 
            default=1.0, type=float)

    args.add_argument("-p", "--predict", help="Generate predictions", 
            metavar="OUTPUT DIR")
    args.add_argument("-t", "--train", help="TRAINING TIME!", action="store_true")

    args.add_argument("--pipe_test", nargs="*", metavar="ARG", 
            help="Test input pipe. Optional args: [1: fraction of input data to transform,\
                    2: output directory]")

    FLAGS, unknown = args.parse_known_args()
    print(FLAGS)
    if FLAGS.train:
        with Trainer(FLAGS.dataset, FLAGS.checkpoint, FLAGS.logs,
                FLAGS.tblogs) as trainer:
            trainer.train(FLAGS.frac)
        
    elif FLAGS.predict is not None:
        with Trainer(FLAGS.dataset, FLAGS.checkpoint, FLAGS.logs,
                FLAGS.tblogs) as trainer:
            trainer.predict(FLAGS.predict)
    elif FLAGS.pipe_test is not None:
        _args = dict()
        if len(FLAGS.pipe_test) >= 1:
            f = float(FLAGS.pipe_test[0])
            f = 0 if f < 0 else 1 if f > 1 else f
            _args['frac'] = f
        if len(FLAGS.pipe_test) >= 2:
            _args['out_dir'] = FLAGS.pipe_test[0]

        with Trainer(FLAGS.dataset, FLAGS.checkpoint, FLAGS.logs,
                FLAGS.tblogs) as trainer:
            trainer.test_input_pipe(**_args)
    else:
        args.print_help()


