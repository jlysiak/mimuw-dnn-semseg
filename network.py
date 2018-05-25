"""
Deep Neural Networks @ MIMUW 2017/18
Jacek Łysiak

Semantic segmentation with convolutional network

"""
import tensorflow as tf
import argparse

import trainer
import dnn_conf

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Deep convolutional network\
            for semantic segmentation - Deep Neural Networks @ MIMUW 2017")

    # ==== MODIFY DEFAULT CONFIGURATION
    args.add_argument("-d", "--dataset", help="Dataset location", metavar="DIR")
    args.add_argument("-c", "--checkpoint", help="Checkpoint directory")
    args.add_argument("-l", "--logs", help="Log file path")
    args.add_argument("-b", "--tblogs", help="Directory for TensorBoard logs")
    args.add_argument("-r", "--lrate", help="Learning rate", type=float)
    args.add_argument("-f", "--frac", help="Fraction of dataset taken.", type=float)
    args.add_argument("--gpu", help="Use GPU", action="store_true")

    # ==== CREATE/LOAD CONFIGURATION
    args.add_argument("-C", "--config", help="Path to configuration file")
    args.add_argument("--new_config", help="Create new default configuration file", 
            nargs="?", const=dnn_conf.DEFAULT_CONF_PATH)
    
    # ==== ACTIONS 
    args.add_argument("--train_pipe_test", metavar="OUT", nargs="?", 
            const="train_pipe_test", help="Test input pipe and generate \
                    imgs into `OUT` dir.")
    args.add_argument("--pred_pipe_test", metavar="OUT", nargs="?", 
            const="pred_pipe_test", help="Test input pipe and generate \
                    imgs into `OUT` dir.")
    
    args.add_argument("--predict", help="Generate predictions", 
            metavar="OUTPUT DIR")
    args.add_argument("--train", help="TRAINING TIME!", action="store_true")
    FLAGS, unknown = args.parse_known_args()
    print(FLAGS)

    if FLAGS.new_config is not None:
        dnn_conf.create_default(FLAGS.new_config)
        print("Network config created: " + FLAGS.new_config)
        exit(0)
    
    if not FLAGS.train and FLAGS.predict is None \
        and FLAGS.train_pipe_test is None and FLAGS.pred_pipe_test is None:
        args.print_help()
        exit(0)

    # Load configuration
    conf_path = dnn_conf.DEFAULT_CONF_PATH
    if FLAGS.config is not None:
        conf_path = FLAGS.config

    config = dnn_conf.load_conf(conf_path)
    
    # Update configuration using passed flags
    if FLAGS.dataset is not None:
        config["DATASET_DIR"] = FLAGS.dataset
    if FLAGS.checkpoint is not None:
        config["CKPT_DIR"] = FLAGS.checkpoint
    if FLAGS.logs is not None:
        config["LOG_PATH"] = FLAGS.logs
    if FLAGS.tblogs is not None:
        config["TB_DIR"] = FLAGS.tblogs
    if FLAGS.lrate is not None:
        config["LEARNING_RATE"] = FLAGS.lrate
    if FLAGS.frac is not None:
        config["DS_FRAC"] = FLAGS.frac
    if FLAGS.gpu:
        config["DEVICE"] = "GPU"

    # Run Trainer
    with trainer.Trainer(config) as T:
        if FLAGS.train:
            T.train()
            
        elif FLAGS.predict is not None:
            T.predict(FLAGS.predict)

        elif FLAGS.train_pipe_test is not None:
            T.test_train_pipe(out_dir=FLAGS.train_pipe_test)
        
        elif FLAGS.pred_pipe_test is not None:
            T.test_pred_pipe(out_dir=FLAGS.pred_pipe_test)


