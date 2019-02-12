'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
import random
import time
import reader
import math

flags = tf.flags
flags.DEFINE_string("vocab_file", "./RNN/data/vocab.csv",
    "File containing the vocabulary")
FLAGS = flags.FLAGS

def pad_up_to(tensor, max_in_dims):
    s = tf.shape(tensor)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(tensor, paddings, "CONSTANT")

class RNNModel(object):
    """model"""
    def __init__(self, vocab_size, config, num_train_samples, num_valid_samples):
        self.vocab_size = vocab_size
        self.batch_size = config.batch_size
        self.max_epoch = config.max_epoch
        self.max_max_epoch = config.max_max_epoch
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self._rnn_params = None
        self.keep_prob = config.keep_prob
        self._cell = None
        self.num_layers = config.num_layers
        self.num_steps = config.num_steps
        self.max_gradient_norm= config.max_grad_norm
        self.learning_rate = config.learning_rate
        self.lr_decay = config.lr_decay
        size = config.hidden_size

        # We set a dynamic learining rate, it decays every time the model has gone through 150 batches.
        # A minimum learning rate has also been set.

        self.file_name_train = tf.placeholder(tf.string)
        self.file_name_validation = tf.placeholder(tf.string)
        self.file_name_test = tf.placeholder(tf.string)

        self.train_epoch = (self.num_train_samples - 1) // self.num_steps

        self.valid_epoch = (self.num_valid_samples - 1) // self.num_steps


        def parse(line):
            line_split = tf.string_split([line])
            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            output_seq = [tf.string_to_number(line_split.values[-1], out_type=tf.int32)]
            return input_seq, output_seq

        training_dataset = tf.data.TextLineDataset(self.file_name_train).map(parse).padded_batch(config.batch_size, padded_shapes=([None], [None]))
        # gets next batch so that wen can combine the two datasets into one

        validation_dataset = tf.data.TextLineDataset(self.file_name_validation).map(parse).padded_batch(config.batch_size, padded_shapes=([None], [None]))

        test_dataset = tf.data.TextLineDataset(self.file_name_test).map(parse).batch(1)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
            training_dataset.output_shapes)


        self.training_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)

        # self.input_batch & self.output_batch may be tensors of shape [batch_size, sequence_length]
        # sequence_length refers to length of sentence + padding
        self.input_batch, self.output_batch = iterator.get_next()

        # inputs = [1, 1, 20]
        self.inputs = pad_up_to(self.input_batch, [1,20])
        self.inputs = tf.reshape(self.inputs, [1, 1, 20])
        self.inputs = tf.cast(self.inputs, tf.float32)

        non_zero_weights = tf.sign(self.input_batch)

        self.valid_words = tf.reduce_sum(non_zero_weights)

        def make_cell():
            cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)
            if config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cll, output_keep_prob=config.keep_prob)
            return cell


        cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        state = self.initial_state


        stop = False

        # Comput sequence length
        def get_length(non_zero_place):
            real_length = tf.reduce_sum(non_zero_place, 1)
            real_length = tf.cast(real_length, tf.int32)
            return real_length

        batch_length = get_length(non_zero_weights)

        # output = [batch_size, max_time_nodes, output_vector_size]
        with tf.variable_scope("RNN"):
            output, state = tf.nn.dynamic_rnn(cell=cell, inputs=self.inputs,
            sequence_length=batch_length, initial_state= state, dtype=tf.float32)

        output = tf.reshape(output,[1, size])

        # Compute logits
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])
        self.probas = tf.nn.softmax(logits, name='p')

        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
            tf.reshape(self.output_batch, [1]), logits = logits)# * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)
        """
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
            targets = self.output_batch,
            weights = tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=False)

        self.loss = loss
        self.cost = tf.reduce_sum(loss)
        self.final_state = state

        # Train
        self.learning_rate = tf.Variable(0.0, trainable=False)
        params = tf.trainable_variables()

        #optimizer = tf.train.GradientDescentOptimizer(self._lr)
        opt= tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10e-4)
        gradients = tf.gradients(self.cost, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params),
            global_step=tf.train.get_or_create_global_step())

    def assign_lr(self, lr_value):
        return tf.assign(self.learning_rate, lr_value)

    def batch_train(self, sess, saver, config, train_file, valid_file):
        """Runs the model on the given data."""
        for i in range(self.max_max_epoch):
            lr_decay = self.lr_decay ** max(i + 1 - self.max_epoch, 0.0)
            self.learning_rate = self.assign_lr(config.learning_rate * lr_decay)
            sess.run(self.training_init_op, {self.file_name_train: "RNN/data/train.txt.ids"})
            state = sess.run(self.initial_state)
            costs = 0.0
            iters = 0
            for step in range(self.train_epoch):
                start_time = time.time()
                cost, current_learning_rate, final_state, _, probas, loss, cost, input= sess.run(
                    [self.cost, self.learning_rate, self.final_state, self.updates, self.probas, self.loss, self.cost, self.input_batch])
                costs += cost

                print("NEW BATCH:")
                print(input)
                print(np.argmax(probas))
                print(loss)
                print(cost)

                iters += self.num_steps
                if step % (self.train_epoch // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step * 1.0 /self.train_epoch, np.exp(costs/iters),
                        iters * self.batch_size / (time.time() - start_time)))

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(self.learning_rate)))
            sess.run(self.validation_init_op, {self.file_name_validation:"RNN/data/valid.txt.ids"})
            dev_costs = 0.0
            state = sess.run(self.initial_state)
            iters = 0
            for step in range(self.valid_epoch):
                start_time = time.time()
                dev_cost, final_state = sess.run(
                    [self.cost, self.final_state])

                dev_costs += dev_cost
                iters += self.num_steps
                if step % (self.valid_epoch // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step % 1.0/self.valid_epoch, np.exp(dev_costs/iters),
                        iters * self.batch_size /(time.time() - start_time)))

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(self.learning_rate)))
            saver.save(sess, "RNN/models/best_model.ckpt")

    def predict(self, sess, input_file, raw_file, verbose=False):
        # if verbose is true, then we print the ppl of every sequence

        sess.run(self.test_init_op, {self.file_name_test: input_file})

        with open(raw_file) as fp:

            global_dev_loss = 0.0

            for raw_line in fp.readlines():

                raw_line = raw_line.strip()

                dev_cost, input_line = sess.run(
                    [self.cost, self.input_batch])



                dev_costs += dev_cost

                if verbose:
                    dev_ppl = np.exp(dev_cost)
                    print(raw_line + " Test PPL: %.3f" % (dev_ppl))

            dev_ppl = math.exp(dev_costs)
            print("Global Test PPL: %.3f" % (dev_ppl))
