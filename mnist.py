"""
Deep Neural Networks @ MIMUW 2017/18
Jacek Łysiak

MNISTrainer

"""
import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
import matplotlib.pyplot as plt 
import shutil
import datetime

# =============================================================================
# ================ MNIST DEEEEP NEURAL NETWORK CONFIGURATION ==================

# Put into LAYERS_CONF 3-tuples: ( layer_type, type_arg, act_fn)
# where layer_type:
#   CONVOLUTIONAL   => 'c', number of features, activation function
#   POOLING         => 'p', None, activation function
#   FULLY CONNECTED => 'f', number of features, activation function
#
# NOTES:
#   1. Only convs 2D perserving size are used
#   2. All convs has spacial dimensions of 3x3
#   3. All poolngs are of type MAX with 2x2 spacial dimesnions and the same stride

LAYERS_CONF = [
    ('c', 32, tf.nn.relu),
    ('p', None , None),
    ('c', 64, tf.nn.relu),
    ('p', None, None),
    ('f', 1024, tf.nn.relu),
    ('f', 10, None)
]

# =============================================================================
# ==================                 STOP                     =================
# =============================================================================

class MnistTrainer(object):

    def __init__(self, layers_conf, mnist_path, ckpt_path=None, 
            img_path=None, log_path="/tmp/mnist.log"):
        if ckpt_path is None:
            self.ckpt_dir = "/tmp"
        else:
            self.ckpt_dir = ckpt_path
        self.ckpt_path = os.path.join(self.ckpt_dir, "mnist.ckpt")
        if img_path is None:
            self.imgs_path = "/tmp"
        else:
            self.imgs_path = img_path
        self.log_path = log_path

        self.layers_conf = layers_conf
        self.mnist_path = mnist_path

    # ====== Window transformations helpers

    def app_pool(self, wnd_fn):
        def n(arg):
            wnd = wnd_fn(arg)
            return (2*wnd[0], 2*wnd[1], 2 * wnd[2], 2*wnd[3])

        return lambda arg: n(arg)

    def app_conv(self, wnd_fn):
        def n(arg):
            wnd = wnd_fn(arg)
            return (wnd[0]-1, wnd[1] + 1, wnd[2] - 1, wnd[3] + 1)
        return lambda arg: n(arg)

    def id(self):
        return lambda x: x

    # ====== Network builders

    def weight_variable(self, shape):
        try:
            stddev = np.prod(shape) ** (-0.5)
        except:
            stddev = np.prod(shape).value ** (-0.5)
        initializer = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initializer, name='weight')

    def batch_norm(self, x, act_fn=None):
        rank = len(x.shape)
        if rank == 2:
            axis = [0]
        elif rank == 4:
            axis = [0, 1, 2]
        else:
            raise Exception("Wrong tensor rank!")
        scale = self.weight_variable(x.shape[1:]) 
        shift = self.weight_variable(x.shape[1:]) 
        m = tf.reduce_mean(x, axis=axis, keep_dims=True)
        var = tf.reduce_mean(tf.square(x - m), axis=axis, keep_dims=True)
        std = tf.sqrt(var)
        x_n = scale * ((x - m) / std) - shift
        if act_fn is not None:
            x_n = act_fn(x_n)
        return x_n

    def create_convnet(self):
        print("Building MNIST convolutional NN...")
        
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                    strides=[1, 2, 2, 1], padding='SAME')

        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y_target = tf.placeholder(tf.float32, [None, 10])
        x = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        
        signal = x
        last = 'c', [28, 28, 1], None
        last_fn = self.id()
        
        conv_layers = []
        funs = []
        
        i = 0
        for ident, arg, act_fn in self.layers_conf:
            print(i, ident, arg, act_fn)
            if ident == 'p':
                signal = max_pool(signal)
                last_fn = self.app_pool(last_fn)

            elif ident == 'c':
                s = [3, 3] + [signal.shape[-1].value] + [arg]
                w = self.weight_variable(s)
                signal = conv2d(signal, w)
                signal = self.batch_norm(signal, act_fn)
                conv_layers.append(signal)
                last_fn = self.app_conv(last_fn)
                funs.append(last_fn)
            
            elif ident == 'f':
                n = np.prod(signal.shape[1:])
                if last[0] != 'f':
                    signal = tf.reshape(signal, shape=[-1, n])
                w = self.weight_variable([n.value, arg])
                signal = tf.matmul(signal, w)
                signal = self.batch_norm(signal, act_fn)
            else:
                raise Exception("Unrecognized layer type!")
            last = ident, arg, act_fn

        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=signal, 
                    labels=self.y_target))
        self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(self.y_target, axis=1), 
                        tf.argmax(signal, axis=1)), 
                    tf.float32
                )
            )
        self.train_step = tf.train.MomentumOptimizer(0.02, momentum=0.9).minimize(self.loss)
        print('list of variables:')
        for name, shape in map(lambda x: (x.name, x.shape), tf.global_variables()):
            print(name, shape)
        self.conv_layers = conv_layers
        self.funs = funs

    # =========== Training and testing 

    def train_on_batch(self, batch_xs, batch_ys):
        self.sess.run([self.train_step],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})

    def train(self):
        self.create_convnet()
        mnist = input_data.read_data_sets(self.mnist_path, one_hot=True)
        results = []
        saver = tf.train.Saver()
        restore = True
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            restore = False

        with tf.Session() as self.sess:
            if restore:
                try:
                    saver.restore(self.sess, self.ckpt_path)
                except:
                    tf.global_variables_initializer().run()
            else:
                tf.global_variables_initializer().run()  # initialize variables
            batches_n = 100000
            mb_size = 128
            losses = []
            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)
                    self.train_on_batch(batch_xs, batch_ys)
                    if batch_idx % 50 == 0:
                        loss, acc =  self.sess.run([self.loss, self.accuracy],
                                feed_dict={self.x: mnist.test.images,
                                    self.y_target: mnist.test.labels})
                        print("Batches passed: {:5} Loss: {:10} Acc: {:10}".format(batch_idx, loss, acc))
                        results.append((batch_idx, loss, acc))

            except KeyboardInterrupt:
                print('Stopping training!')

            print("Model saved as"  + saver.save(self.sess, self.ckpt_path)) 
            if self.log_path is not None:
                with open(self.log_path, 'w') as f:
                    f.write('Creation time: {:%Y-%b-%d %H:%M:%S}\n\n'.format(datetime.datetime.now()))
                    f.write("{:5} {:6} {:6}\n".format("B.IDX", "LOSS", "ACC"))
                    for idx, loss, acc in results:
                        f.write("{:4}: {:<.4f} {:<.4f}\n".format(idx, loss, acc))
                    # Test trained model
                    tloss, tacc = self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                                                    self.y_target: mnist.test.labels})
                    f.write("\n===== TEST RESULTS =======\n")
                    f.write("{:<.4f} {:<.4f}\n".format(tloss, tacc))

    def visualize(self):
        self.create_convnet()
        mnist = input_data.read_data_sets(self.mnist_path, one_hot=True)
        saver = tf.train.Saver()
        if not os.path.exists(self.imgs_path):
            os.makedirs(self.imgs_path)
        with tf.Session() as self.sess:
            try:
                saver.restore(self.sess, self.ckpt_path)
            except:
                print("NO CHECKPOINT FILES IN: " + self.ckpt_path)
                exit(1)

            imgs_n = 2000
            try:
                img_xs = mnist.train.images[:imgs_n]
                img_ys = mnist.train.labels[:imgs_n]
                loss, acc =  self.sess.run([self.loss, self.accuracy],
                                feed_dict={self.x: img_xs, self.y_target: img_ys})
                outs =  self.sess.run(self.conv_layers,
                                feed_dict={self.x: img_xs, self.y_target: img_ys})
                for cidx, el in enumerate(outs):
                    ch_sz = el.shape[-1]
                    rows = ch_sz // 8
                    cols = 8
                    plt.close('all')
                    fig, ax = plt.subplots(cols, rows, sharex=True, sharey=True)
                    fn = self.funs[cidx]
                    for ch_idx in range(ch_sz):
                        ch_max = 0
                        ch_max_xy = None, None
                        ch_max_wnd = None
                        for img_idx in range(imgs_n):
                            img = np.reshape(img_xs[img_idx], (28,28))
                            e = el[img_idx]
                            f = e[:,:,ch_idx]
                            idx = np.unravel_index(np.argmax(f), f.shape)
                            v = f[idx]
                            if v > ch_max:
                                ch_max = v
                                ch_max_xy = idx
                                x, y = idx
                                x1, x2, y1, y2 = fn((x, x+1, y, y + 1))
                                l = [np.clip(k, 0, 28) for k in [ x1, x2, y1, y2]]
                                ch_max_wnd = img[l[0]:l[1], l[2]:l[3]]
                        print("Channel: %d" % ch_idx, ch_max, ch_max_xy) 
                        fx = ch_idx % cols
                        fy = ch_idx // cols
                        ax[fx, fy].matshow(ch_max_wnd, cmap='gray')
                        ax[fx, fy].tick_params(axis='x', which='both', 
                                bottom=False, top=False, 
                                labelbottom=False, labeltop=False)
                        ax[fx, fy].tick_params(axis='y', left=False, 
                                right=False, labelleft=False, labelright=False)
                    fig.savefig(os.path.join(self.imgs_path, 
                        "img-%d.png" % (cidx)))
            except KeyboardInterrupt:
                print('Stopped!')

 
if __name__ == '__main__':
    args = argparse.ArgumentParser(description="MNISTrainer for DNN @ MIMUW 2017")
    args.add_argument("-M", "--mnist", help="MNIST dataset location", 
            metavar="DIR", default="MNIST_data")
    args.add_argument("-c", "--checkpoint", help="Checkpoint directory")
    args.add_argument("-i", "--imgs", help="Images save directory")
    args.add_argument("-t", "--train", help="Run training", action="store_true")
    args.add_argument("-l", "--log", help="Log file path")
    args.add_argument("-v", "--visualize", help="Visualize filters", action="store_true")

    FLAGS, unknown = args.parse_known_args()
    if FLAGS.train:
        trainer = MnistTrainer(LAYERS_CONF, FLAGS.mnist, 
                FLAGS.checkpoint, FLAGS.imgs, FLAGS.log)
        trainer.train()
    elif FLAGS.visualize:
        trainer = MnistTrainer(LAYERS_CONF, FLAGS.mnist, 
                FLAGS.checkpoint, FLAGS.imgs, FLAGS.log)
        trainer.visualize()
    else:
        args.print_help()


