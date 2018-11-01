import reader

import RNN
import os
import tensorflow as tf


'''
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

'''

flags = tf.flags
logging = tf.logging


flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

flags.DEFINE_string("save_path","./models" ,
                    "Model output directory.")
flags.DEFINE_string("train_file", "ptb.train.txt",
                    "The file containing the training data")
flags.DEFINE_string("valid_file", "ptb.valid.txt",
                    "The file containing the validation data")
flags.DEFINE_string("test_file", "ptb.test.txt",
                    "The file containing the testing data")
FLAGS = flags.FLAGS


# Set TRAIN todo true will build a new model
TRAIN = True

# If VERBOSE is true, then print the ppl of every sequence when we are testing
VERBOSE = True

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 1
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 1
    vocab_size = 10000

def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

    return config


# generate vocabulary and ids for all data
if not os.path.isfile("data/vocab.tx"):
    reader.gen_vocab(FLAGS.train_file)
if not os.path.isfile("data/" + FLAGS.train_file + ".ids"):
    reader.gen_id_seqs(FLAGS.train_file)
    reader.gen_id_seqs(FLAGS.valid_file)

with open("data/" + FLAGS.train_file + ".ids") as fp:
    num_train_samples = len(fp.readlines())
with open("data/" + FLAGS.valid_file + ".ids") as fp:
    num_valid_samples = len(fp.readlines())

with open("data/vocab.txt") as vocab:
    vocab_size = len(vocab.readlines())

config = get_config()

def create_model(sess):
    model = RNN.RNNModel(vocab_size = vocab_size, config=config,
        num_train_samples=num_train_samples, num_valid_samples=num_valid_samples)
    sess.run(tf.global_variables_initializer())
    return model

if TRAIN:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = create_model(sess)
        saver = tf.train.Saver()
        model.batch_train(sess, saver, config)

tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = create_model(sess)
    saver = tf.train.Saver()
    saver.restore(sess, "model/best_model.ckpt")
    predict_id_file = os.path.join("data/" + FLAGS.test_file + ".ids")
    if not os.path.isfile(predict_id_file):
        gen_id_seqs(test_file)
    model.predict(sess,predict_id_file, test_file, verbose=VERBOSE)
