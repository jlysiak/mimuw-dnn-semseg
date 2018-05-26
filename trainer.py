import tensorflow as tf
import numpy as np
import math
import os
import shutil
import datetime
import sys
from PIL import Image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

import net_builder
import pipe_train

class CONF:
    def __init__(self, kw):
        self.__dict__ = kw


class Trainer(object):
    """
    Main trainer class
    """
    
    def __init__(self, config):
        conf = CONF(config)

        if not os.path.exists(conf.DATASET_DIR):
            raise Exception("Dataset directory does not exist!")

        config['DATASET_IMAGES'] = os.path.join(conf.DATASET_DIR, conf.IMAGES_DIR)
        config['DATASET_LABELS'] = os.path.join(conf.DATASET_DIR, conf.LABELS_DIR)
        if not os.path.exists(conf.CKPT_DIR):
            os.makedirs(conf.CKPT_DIR)
        config['CKPT_PATH'] = os.path.join(conf.CKPT_DIR, conf.CKPT_NAME)
        
        stamp = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
        config['START_TIMESTAMP'] = stamp

        self.log_file = open(conf.LOG_PATH, "a")
        self.t_tb_writer = tf.summary.FileWriter(
                os.path.join(conf.TB_DIR, stamp, "train"))
        self.v_tb_writer = tf.summary.FileWriter(
                os.path.join(conf.TB_DIR, stamp, "valid"))
        self.config = config


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.log_file.close()


    def log(self, s):
        timestamp =  '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        s = "[{:>19s}] {}\n".format(timestamp, s)
        sys.stderr.write(s)        
        self.log_file.write(s)

        
    def setup_pipe(self, is_training_phase=True):
        """
        Wraper for input pipe creator.
        Places pipe on cpu in proper name scope.
        """
        conf = CONF(self.config)
        log = lambda x: self.log(x)

        with tf.device("/device:CPU:0"):
            with tf.name_scope("input_pipe"):
                if is_training_phase:
                    return pipe_train.setup_train_valid_pipes(conf, log)


    def test_pred_pipe(self, 
            out_dir="pipe_test", 
            train_outs="training", 
            valid_outs="validation"):
        pass

    def test_train_pipe(self, 
            out_dir="pipe_test", 
            train_outs="training", 
            valid_outs="validation"):
        """
        Handy method to check input pipe.
        """
        conf = CONF(self.config)
        log = lambda x: self.log(x)
        
        t_outs = os.path.join(out_dir, train_outs)
        v_outs = os.path.join(out_dir, valid_outs)
        if not os.path.exists(t_outs):
            os.makedirs(t_outs)
        if not os.path.exists(v_outs):
            os.makedirs(v_outs)

        log("******* Input pipe test")
        it_t, it_v = self.setup_pipe()
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
        conf = CONF(self.config)
        log = lambda x: self.log(x)
        
        log("**** BUILDING NETWORK")
        
        #train_indicator = tf.placeholder_with_default(
        #        input=True, 
        #        shape=[0],
        #        name="TRAIN_INDICATOR")
        train_indicator = tf.placeholder(
                dtype=tf.bool,
                name="TRAIN_INDICATOR")

        dev = "/device:%s:0" % conf.DEVICE 
        with tf.device(dev):
            with tf.name_scope("network"):
                # Network builder chnges data format if needed
                # but returns NHWC
                logits = net_builder.build_network(
                            x=x_iter,
                            batch_size=conf.BATCH_SZ,
                            arch_list=conf.ARCHITECTURE,
                            log=log,
                            train_indicator=train_indicator,
                            dfmt=conf.DATA_FORMAT)

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
            train_step = tf.train.RMSPropOptimizer(conf.LEARNING_RATE, 
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
        # PHASE INDICATOR 
        self.indicator = train_indicator


    def train(self):
        """
        Train model.
        Args:
            [files_frac]: fraction of files used in training
        """
        conf = CONF(self.config)
        log = lambda x: self.log(x)
        
        log("******* Starting training procedure...")
        it_t, it_v = self.setup_pipe()
        next_t = it_t.get_next()
        next_v = it_v.get_next()
        
        self.build_network(next_t[0], next_t[1])
        
        saver = tf.train.Saver()
        restore = True
        if not os.path.exists(conf.CKPT_DIR):
            os.makedirs(conf.CKPT_DIR)
            restore = False
        
        config = tf.ConfigProto(
		allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            log("Initialize training dataset...")
            it_t.initializer.run()

            if restore:
                try:
                    log("Attempt to restore model from: %s" % conf.CKPT_PATH)
                    saver.restore(sess, conf.CKPT_PATH)
                    log("DONE!")
                except:
                    log("Cannot restore... Initializing new variables")
                    tf.global_variables_initializer().run()
            else:
                log("Initializing new network variables")
                tf.global_variables_initializer().run()

            epochs = 1
            batch_n = 0
            feed = {self.indicator: True}
            log("**** TRAINING STARTED")
            try:
                for epoch in range(epochs):
                    log("** Epoch: %d" % (epoch + 1))
                    try:
                        while True:
                            # Feed training indicator 
                            t_loss, t_acc, _ = sess.run([
                                self.t_loss, 
                                self.t_acc, 
                                self.t_step],
                                feed_dict=feed)
                            batch_n += 1
                            log("batch: %d, loss: %f, accuracy: %1.3f" % 
                                        (batch_n, t_loss, t_acc))

                            if batch_n % 500 == 0:
                                t_loss, t_acc, t_summ, _ = sess.run([
                                    self.t_loss, 
                                    self.t_acc, 
                                    self.t_summaries, 
                                    self.t_step],
                                    feed_dict=feed)
                                self.t_tb_writer.add_summary(t_summ, batch_n)
                                batch_n += 1
                                log("batch: %d, loss: %f, accuracy: %1.3f" % 
                                            (batch_n, t_loss, t_acc))
                    
                    except tf.errors.OutOfRangeError:
                        log("** End of dataset!")
            except KeyboardInterrupt:
                log("Training stopped by keyboard interrupt!")
            
            save_path = saver.save(sess, conf.CKPT_PATH)
            log("Model saved as: %s" % save_path)

    def predict(self, outdir):
        pass
