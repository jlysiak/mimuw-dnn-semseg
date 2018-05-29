import tensorflow as tf
from .utils import mkflags
from .builder import network_builder

def build_network(config):
    """
    Build neural network.
    """
    FLAGS = mkflags(config)

    # ==== PLACEHOLDERS
    # I didn't have much more time to upgrade my 
    # solution to shared variables model...
    x_ph = tf.placeholder(dtype=tf.float32, name="INPUT")
    y_ph = tf.placeholder(dtype=tf.int32, name="LABELS")
    ind_ph = tf.placeholder(dtype=tf.bool, name="TRAIN_INDICATOR")
    x = x_ph
    y = y_ph

    dev = "/device:%s:0" % FLAGS.DEVICE 
    with tf.device(dev):
        with tf.name_scope("network"):
            # Network builder chnges data format if needed
            # but returns NHWC
            logits, layers = network_builder(x=x, batch_size=FLAGS.BATCH_SZ,
                        arch_list=FLAGS.ARCHITECTURE, train_indicator=ind_ph,
                        dfmt=FLAGS.DATA_FORMAT)

    pred = tf.argmax(logits, axis=3, output_type=tf.int32)
    _orig_sz = [FLAGS.PREDICTION_SZ] * 2
    pred_orig = tf.reshape(pred, shape=[-1] + pred.shape.as_list()[1:3] + [1])
    pred_orig = tf.image.resize_images(images=pred_orig, 
            size=_orig_sz, 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    pred_orig = tf.reshape(pred_orig, shape=[-1] + _orig_sz)
    
    # Save one image, label and prediction from each batch
    t_summaries = []
    t_summaries += [tf.summary.image("train/imgs/label", 
        tf.cast(y, tf.uint8), max_outputs=1)]
    t_summaries += [tf.summary.image("train/imgs/image", 
        x, max_outputs=1)]
    t_summaries += [tf.summary.image("train/imgs/prediction", 
        tf.cast(tf.reshape(pred, [-1, FLAGS.INPUT_SZ, FLAGS.INPUT_SZ, 1]), 
        tf.uint8), max_outputs=1)]
    
    y = tf.reshape(y, [-1, FLAGS.INPUT_SZ, FLAGS.INPUT_SZ])
    
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
    # Add batch loss and accuracy
    t_summaries += [tf.summary.scalar("train/acc", acc)]
    t_summaries += [tf.summary.scalar("train/loss", loss)]
    
    # TRAINING OPS - within batch
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.RMSPropOptimizer(FLAGS.LEARNING_RATE,
               momentum=0.9).minimize(loss)
    
    # VALIDATION OPS - cumulative across dataset
    # We can take mean of means over batches beacuse 
    # all batches has same size.
    acc_cum, acc_cum_update = tf.metrics.mean(
            values=acc,
            name="valid/metrics/acc")
    loss_cum, loss_cum_update = tf.metrics.mean(
            values=loss,
            name="valid/metrics/loss")
    v_summaries = [] 
    v_summaries += [tf.summary.scalar("valid/loss", loss_cum)]
    v_summaries += [tf.summary.scalar("valid/acc", acc_cum)]
    
    metrics_vars = []
    for el in tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES,
            scope="valid/metrics"):
        metrics_vars.append(el)
    metrics_init = tf.variables_initializer(var_list=metrics_vars)

    update = [acc_cum_update, loss_cum_update]
    t_summaries = tf.summary.merge(t_summaries)
    v_summaries = tf.summary.merge(v_summaries)
    
    extra = {
        'PRED': pred,
        'PRED_ORIG': pred_orig,
        'TRAIN': train_step,
        'LOSS': loss,
        'ACC': acc,
        'LOSS_CUM': loss_cum,
        'ACC_CUM': acc_cum,
        'METRICS_INIT': metrics_init,
        'METRICS_UPDATE': update,
        'T_SUMMARY': t_summaries,
        'V_SUMMARY': v_summaries,
        'LAYERS': layers
    }
    return x_ph, y_ph, ind_ph, extra 
