import tensorflow as tf
from .utils import mkflags
from .pipe_train import setup_train_pipe
from .pipe_valid import setup_valid_pipe
from .pipe_pred import setup_pred_pipe

def setup_pipe(pipe_type, config, imgs, labs=None):
    """
    Build input pipe appropriate to given type
    Args:
        pipe_type: pipe type
        config: configuration dict
        imgs: images/examples
        labels: 
    """
    with tf.device("/device:CPU:0"):
        with tf.name_scope("input_pipe"):
            if pipe_type == "training":
                if labs is None:
                    raise Exception("Training pipe requires labels!")
                return setup_train_pipe(config, imgs, labs)

            elif pipe_type == "validation":
                if labs is None:
                    raise Exception("Validation pipe requires labels!")
                return setup_valid_pipe(config, imgs, labs)

            elif pipe_type == "prediction":
                return setup_pred_pipe(config, imgs)

            else:
                raise Exception("Invalid pipe type: %s" % pipe_type)

