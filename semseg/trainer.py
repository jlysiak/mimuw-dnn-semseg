import tensorflow as tf
import numpy as np
import math
import os
import shutil
import datetime
import time
import sys
from random import shuffle

from .utils import mkflags, to_sec, get_image_paths
from .images import save_predictions, save_image
from .pipe import setup_pipe
from .network import build_network

class Trainer(object):
    """
    Main trainer class
    """
    
    def __init__(self, config):
        FLAGS = mkflags(config)

        if not os.path.exists(FLAGS.CKPT_DIR):
            os.makedirs(FLAGS.CKPT_DIR)
        config['CKPT_PATH'] = os.path.join(FLAGS.CKPT_DIR, FLAGS.CKPT_NAME)

        stamp = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
        config['START_TIMESTAMP'] = stamp
        config['TIME_START'] = time.time() 

        self.log_file = open(FLAGS.LOG_PATH, "a")
        self.config = config


    def _init(self):
        """
        Some init stuff before training and validation.
        """
        FLAGS = mkflags(self.config)
        config = self.config
        log = lambda x: self.log(x)

        if not os.path.exists(FLAGS.DATASET_DIR):
            raise Exception("Dataset directory does not exist!")
        
        config['DATASET_IMAGES'] = os.path.join(FLAGS.DATASET_DIR, FLAGS.IMAGES_DIR)
        config['DATASET_LABELS'] = os.path.join(FLAGS.DATASET_DIR, FLAGS.LABELS_DIR)
        
        self.t_tb_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.TB_DIR, FLAGS.START_TIMESTAMP, "train"))
        self.v_tb_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.TB_DIR, FLAGS.START_TIMESTAMP, "valid"))

        time_end = time.time() + to_sec(FLAGS.TIME_LIMIT)
        config['TIME_END'] = time_end
        config['TIMESTAMP_END'] =  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_end))
        if "LOCK" not in config:
            config["LOCK"] = False

        self.config = config
        FLAGS = mkflags(self.config)

        # Get training data
        imgs = os.listdir(FLAGS.DATASET_IMAGES)
        labs = os.listdir(FLAGS.DATASET_LABELS)

        n = int(len(imgs) * FLAGS.DS_FRAC)
        if n < 10:
            raise Exception("Not enough files in dataset...")
        n_v = max(int(n * FLAGS.VALID_DS_FRAC), 2)
        n_v = min(n - 1, n_v)
        n_v = min(n_v, FLAGS.VALID_DS_FILE_LIMIT)
        n_t = n - n_v
        log("Examples in total: %d, train: %d, valid: %d" % (n, n_t, n_v))
        
        # Prevent non-deterministic file listing
        imgs.sort()
        labs.sort()
        _ex = [i for i in zip(imgs, labs)]
        shuffle(_ex)

        imgs = [os.path.join(FLAGS.DATASET_IMAGES, el) for el, _ in _ex[:n]]
        labs = [os.path.join(FLAGS.DATASET_LABELS, el) for _, el in _ex[:n]]
        return imgs, labs, n_t


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.log_file.close()


    def log(self, s):
        timestamp =  '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        elapsed = time.time() - self.config['TIME_START']
        h = int(elapsed // 3600)
        m = int((elapsed // 60) % 60)
        sec = int(elapsed % 60)
        s = "[{:>19s} | {:02d}:{:02d}:{:02d}] {}\n".format(timestamp, h, m, sec,s)
        sys.stderr.write(s)        
        self.log_file.write(s)


    def show_layers(self, layers):
        self.log("Created layers: ")
        for idx, layer in enumerate(layers):
            self.log("%d: %s, %s" % (idx, layer.name, str(layer.shape)))



    def train(self):
        """
        Train model.
        Args:
            [files_frac]: fraction of files used in training
        """
        self._train_init()

        FLAGS = mkflags(self.config)
        log = lambda x: self.log(x)

        # Get training data
        imgs = os.listdir(FLAGS.DATASET_IMAGES)
        labs = os.listdir(FLAGS.DATASET_LABELS)

        n = int(len(imgs) * FLAGS.DS_FRAC)
        if n < 10:
            raise Exception("Not enough files in dataset...")
        n_v = max(int(n * FLAGS.VALID_DS_FRAC), 2)
        n_v = min(n - 1, n_v)
        n_v = min(n_v, FLAGS.VALID_DS_FILE_LIMIT)
        n_t = n - n_v
        log("Examples in total: %d, train: %d, valid: %d" % (n, n_t, n_v))
        
        # Prevent non-deterministic file listing
        imgs.sort()
        labs.sort()
        _ex = [i for i in zip(imgs, labs)]
        shuffle(_ex)

        imgs = [os.path.join(FLAGS.DATASET_IMAGES, el) for el, _ in _ex[:n]]
        labs = [os.path.join(FLAGS.DATASET_LABELS, el) for _, el in _ex[:n]] 
        
        log("** Starting training procedure...")
        t_init, t_next = setup_pipe("training", self.config, 
                imgs=imgs[:n_t], labs=labs[:n_t])
        v_init, v_next = setup_pipe("validation", self.config, 
                imgs=imgs[n_t:], labs=labs[n_t:])
        
        log("** Building network...")
        x_ph, y_ph, ind_ph, extra = build_network(self.config)
        t_loss = extra['LOSS']
        t_acc = extra['ACC']
        t_summaries = extra['T_SUMMARY']
        t_step = extra['TRAIN']

        self.show_layers(extra['LAYERS'])


        saver = tf.train.Saver()
        
        restore = True
        if not os.path.exists(FLAGS.CKPT_DIR):
            os.makedirs(FLAGS.CKPT_DIR)
            restore = False
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if restore:
                try:
                    log("Attempt to restore model from: %s" % FLAGS.CKPT_PATH)
                    saver.restore(sess, FLAGS.CKPT_PATH)
                    log("DONE!")
                except:
                    log("Cannot restore... Initializing new variables")
                    tf.global_variables_initializer().run()
            else:
                log("Initializing new network variables")
                tf.global_variables_initializer().run()
            
            batch_n = 0
            is_over = False

            log("**** TRAINING STARTED")
            log("Training time limit: %s" % FLAGS.TIMESTAMP_END)

            try:
                for epoch in range(FLAGS.EPOCHS):
                    log("Initialize training dataset...")
                    t_init.run()

                    log("** Epoch: %d" % (epoch + 1))
                    try:
                        while True:
                            batch_n += 1
                            x_val, y_val = sess.run(t_next)
                            feed = {
                                x_ph: x_val,
                                y_ph: y_val,
                                ind_ph: True
                            }

                            if batch_n % 500 == 0:
                                loss, acc, summ, _ = sess.run([t_loss, t_acc, 
                                    t_summaries, t_step], feed_dict=feed)
                                self.t_tb_writer.add_summary(summ, batch_n)
                                log("batch: %d, loss: %f, accuracy: %1.3f" % 
                                            (batch_n, loss, acc))
                            else:
                                # Feed training indicator 
                                sess.run(t_step, feed_dict=feed)

                            if batch_n % FLAGS.VALIDATION_IVAL == 0:
                                pass # TODO validation!
                                
                            # Check time limit
                            if time.time() > FLAGS.TIME_END:
                                log("Training interrupted by reaching time limit.")
                                is_over = True
                                break
                    
                    except tf.errors.OutOfRangeError:
                        log("** End of dataset!")

                    # Save model after each epoch
                    if not FLAGS.LOCK:
                        save_path = saver.save(sess, FLAGS.CKPT_PATH)
                        log("Model saved as: %s" % save_path)

                    # Exit when time limit exceeded
                    if is_over:
                        return

            except KeyboardInterrupt:
                log("Training stopped by keyboard interrupt!")
                # Save model when training was interrupted by user 
                if not FLAGS.LOCK:
                    save_path = saver.save(sess, FLAGS.CKPT_PATH)
                    log("Model saved as: %s" % save_path)
    
    def validate(self, sess, it_v_init, it_v_next, x, y, batch_n, trans_num):
        """
        This method is called within train main loop inside keyboard interrupt
        try-catch.
        """
        # Initialize validation pipeline
        log = lambda x: self.log(x)
        it_v_init.run()
        self.v_init.run()
        log("** Begin validation @ %d" % batch_n)
        log("AVG over %d augumentations" % trans_num)
        imgs = []
        i = 0
        ls = 0
        acs = 0
        try:
            while True:
                x_val, y_val = sess.run(it_v_next)
                vfeed = {
                    x: x_val,
                    y: y_val,
                    self.indicator: False
                }
                # Calculate image average
                v_loss, v_acc, _ = sess.run([
                    self.v_loss, 
                    self.v_acc, 
                    self.v_update],
                    feed_dict=vfeed)
                i += 1
                ls += v_loss
                acs += v_acc
                if i == trans_num:
                    imgs.append((ls / trans_num, acs / trans_num))
                    i = 0
                    ls = 0
                    acs = 0
                    log("Image %d: loss=%f acc=%1.4f" % (len(imgs), imgs[-1][0], imgs[-1][1]))
        
        except tf.errors.OutOfRangeError:
            log("** End of validation dataset!")
        # Get cumulative loss and accyract over VALID set
        v_loss, v_acc, v_summ = sess.run([
            self.v_loss_cum, 
            self.v_acc_cum, 
            self.v_summaries],
            feed_dict=vfeed)
        
        self.v_tb_writer.add_summary(v_summ, batch_n)
        log("Validation results after %d batches: loss=%f acc=%1.3f" % 
                (batch_n, v_loss, v_acc))
    
    def predict(self, paths, outdir=None):
        """
        Generate predictions for given `*.jpg` images.
        Args:
            paths: list of files/dirs paths
            outdir: output directory (default: predictions-<timestamp>)
        """
        FLAGS = mkflags(self.config)
        log = lambda x: self.log(x)
        log("*********** PREDICTION GENERATOR")
        log("Generating predictions for:")
        for p in paths:
            log("  " + p)

        paths = get_image_paths(paths)
        if len(paths) == 0:
            print("No images to predict!")
            return

        name = None
        predictions = None
        counters = None

        if not tf.train.checkpoint_exists(FLAGS.CKPT_PATH):
            raise Exception("No valid checkpoint with prefix: %s" % FLAGS.CKPT_PATH)
        
        if outdir is None:
            stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
            outdir = "predictions-" + stamp

        log("Building input pipe...")
        ds_init, ds_next = setup_pipe("prediction", self.config, imgs=paths)

        log("Building network...")
        x_ph, _, ind_ph, extra = build_network(self.config)
        y_pred = extra['PRED_ORIG']
        wnd_ph = extra['ORIG_SZ']
        self.show_layers(extra['LAYERS'])

        saver = tf.train.Saver()
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            try:
                log("Loading checkpoint from:  %s" % FLAGS.CKPT_PATH)
                saver.restore(sess, FLAGS.CKPT_PATH)
                log("DONE!")
            except:
                raise Exception("Cannot restore checkpoint: %s" % FLAGS.CKPT_PATH)

            if not os.path.exists(outdir):
                os.makedirs(outdir)
                log("Created new directory for predictions: " + outdir)

            sess.run(ds_init)
            i = 0
            j = 0
            try:
                while True:
                    _img_crop, _name, _shape, _wnd = sess.run(ds_next)
                    if name != _name:
                        saved_name = save_predictions(outdir, name, predictions, counters)
                        if saved_name is not None:
                            i += 1
                            log("Prediction %d saved at %s" % (i, saved_name))
                        j = 0
                        name = _name[0]
                        predictions = np.zeros(_shape)
                        counters = np.zeros(_shape)
                        log("Generating prediction for %s [%d x %d]" % 
                                (name, _shape[1], _shape[0]))
                    x1 = _wnd[1]
                    x2 = _wnd[3] + 1
                    y1 = _wnd[0]
                    y2 = _wnd[2] + 1
                    w = x2 - x1 
                    h = y2 - y1
                    feed = {
                        ind_ph: False,
                        x_ph: _img_crop,
                        wnd_ph: [h, w]
                    }
                    _pred_crop = sess.run(y_pred, feed_dict=feed)
                    j += 1
                    log("Image: {} (part: {}) wnd: [{} {} {} {}]".format(i, j, x1, y1, x2, y2)) 
                    predictions[y1:y2, x1:x2] += _pred_crop[0]
                    counters[y1:y2, x1:x2] += 1

            except tf.errors.OutOfRangeError:
                log("End of dataset!")
                save_predictions(outdir, name, predictions, counters)

            except KeyboardInterrupt:
                log("Interrupted by user...")


    def test_valid_pipe(self, v_outs="test_pipe_valid"):
        FLAGS = mkflags(self.config)
        log = lambda x: self.log(x)

        imgs, labs, n_t = self._init()
        v_init, v_next = setup_pipe("validation", self.config, 
                imgs=imgs[n_t:], labs=labs[n_t:])

        if not os.path.exists(v_outs):
            os.makedirs(v_outs)

        with tf.Session() as sess:
            sess.run(v_init)
            try:
                i = 0
                while True:
                    img, lab, name, sh, wnd = sess.run(v_next)
                    print(img.shape, lab.shape, name, sh, wnd)
                    save_image(v_outs, "%d-a.jpg" % i, img)
                    save_image(v_outs, "%d-a.png" % i, np.squeeze(lab))
                    i += 1
            except tf.errors.OutOfRangeError:
                print("End...")
            except KeyboardInterrupt:
                log("Stopped by keyboard interrupt!")

