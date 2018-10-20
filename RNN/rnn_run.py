'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.

Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import collections
import time

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path","./models" ,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 0,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

# Text file containing words for training
training_file = 'ptb.train.txt'




# number of units in RNN cell

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
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


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
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


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
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK

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
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config



def export_state_tuples(state_tuples, name):
  for state_tuple in state_tuples:
    tf.add_to_collection(name, state_tuple.c)
    tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tuples, name, num_replicas):
  restored = []
  for i in range(len(state_tuples) * num_replicas):
    c = tf.get_collection_ref(name)[2 * i + 0]
    h = tf.get_collection_ref(name)[2 * i + 1]
    restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
  return tuple(restored)


def with_prefix(prefix, name):
  """Adds prefix to name."""
  return "/".join((prefix, name))

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

class Input(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = shape_dataset(data, batch_size, num_steps, name)

def shape_dataset(data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "DataShaper", [data, batch_size, num_steps]):
        data = tf.convert_to_tensor(data, name="data", dtype=tf.int32)

        data_len = tf.size(data)
        batch_len = data_len // batch_size
        data = tf.reshape(data[0 : batch_size * batch_len],
            [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message= "epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epcoh_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i*num_steps],
            [batch_size, (i+1)*num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
            [batch_size, (i+1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

class RNNModel(object):
    """model"""
    def __init__(self, is_training, config, input_, vocab_size):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = vocab_size

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b =tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
            config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and  config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state
        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs,
        #                                   initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def _get_lstm_cell(self, config, is_training):
        return tf.contrib.rnn.LSTMBlockCell(
            config.hidden_size, forget_bias=0.0)

    def assign_lr(self, sess, lr_value):
        sess.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections"""
        self._name = name
        ops = {with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = with_prefix(self._name, "initial")
        self._final_state_name = with_prefix(self._name, "final")
        export_state_tuples(self._initial_state, self._initial_state_name)
        export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Import ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                rnn_params,
                base_variable_scope="Model/RNN")
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS,params_saveable)
        self._cost =tf.get_collection_ref(with_prefix(self._name, "cost"))[0]
        num_replicas = 0
        self._initial_state = import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
      return self._input

    @property
    def initial_state(self):
      return self._initial_state

    @property
    def cost(self):
      return self._cost

    @property
    def final_state(self):
      return self._final_state

    @property
    def lr(self):
      return self._lr

    @property
    def train_op(self):
      return self._train_op

    @property
    def initial_state_name(self):
      return self._initial_state_name

    @property
    def final_state_name(self):
      return self._final_state_name

def run_epoch(sess, model, eval_op=None, verbose=False):
    """Rns the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = sess.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 /model.input.epoch_size, np.exp(costs /iters),
                iters * model.input.batch_size / (time.time() - start_time)))
    return np.exp(costs/iters)


def main():
    data = read_data(training_file)
    print("Loaded training data...")

    dictionary, reverse_dictionary = build_dataset(data)
    vocab_size = len(dictionary)
    training_data = [dictionary[word] for word in data if word in dictionary]

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    writer = tf.summary.FileWriter(FLAGS.save_path)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.name_scope("Train"):
            train_input = Input(config = config, data=training_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
                m = RNNModel(is_training=True, config = config, input_= train_input, vocab_size=vocab_size)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = Input(config=config, data=training_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
                mvalid = RNNModel(is_training=False, config=config, input_= valid_input, vocab_size=vocab_size)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = Input(config=eval_config, data=training_data, name="TestInput")
            with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
                mtest = RNNModel(is_training=False, config=eval_config, input_=test_input, vocab_size=vocab_size)

        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        soft_placement = False

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as sess:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(sess, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
                train_perplexity = run_epoch(sess, m, eval_op=m.train_op,
                    verbose=True)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, train_perplexity))
            sv.saver.save(sess, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
    main()
    """
    # tf Graph input
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, vocab_size])

    # RNN output node weights and biases
    weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    biases = tf.Variable(tf.random_normal([vocab_size]))

    saver = tf.train.Saver()

    pred = RNN(x, weights, biases)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0,n_input+1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        while step < training_iters:
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(training_data)-end_offset):
                offset = random.randint(0, n_input+1)

            symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            loss_total += loss
            acc_total += acc
            if (step+1) % display_step == 0:
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
            step += 1
            offset += (n_input+1)
        saver.save(session, logs_path + "model")

        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % (logs_path))
        print("Point your web browser to: http://localhost:6006/")
        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)

            except:
                print("Word not in dictionary")
        """
