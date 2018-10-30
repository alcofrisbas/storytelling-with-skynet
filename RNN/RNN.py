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
import collections
import time
import reader
import math



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
        self._cell = None
        self.num_layers = config.num_layers
        self.num_steps = config.num_steps
        self.max_gradient_norm= config.max_grad_norm
        self.learning_rate = config.learning_rate
        self.lr_decay = config.lr_decay
        size = config.hidden_size

        # We set a dynamic learining rate, it decays every time the model has gone through 150 batches.
        # A minimum learning rate has also been set.
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.file_name_train = tf.placeholder(tf.string)
        self.file_name_validation = tf.placeholder(tf.string)
        self.file_name_test = tf.placeholder(tf.string)

        self.train_epoch = (self.num_train_samples - 1) // self.num_steps
        self.valid_epoch = (self.num_valid_samples - 1) // self.num_steps

        def parse(line):
            line_split = tf.string_split([line])
            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
            return input_seq, output_seq

        training_dataset = tf.data.TextLineDataset(self.file_name_train).map(parse).padded_batch(config.batch_size, padded_shapes=([None],[None]))
        validation_dataset = tf.data.TextLineDataset(self.file_name_validation).map(parse).padded_batch(config.batch_size, padded_shapes=([None],[None]))
        test_dataset = tf.data.TextLineDataset(self.file_name_test).map(parse).batch(1)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
            training_dataset.output_shapes)

        self.input_batch, self.output_batch = iterator.get_next()

        self.training_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)

        # input embedding
        embedding = tf.get_variable(
            "embedding", [self.vocab_size, size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self.input_batch)

        non_zero_weights = tf.sign(self.input_batch)
        self.valid_words = tf.reduce_sum(non_zero_weights)

        # Compute sequence length
        def get_length(non_zero_place):
            real_length = tf.reduce_sum(non_zero_place, 1)
            real_length = tf.cast(real_length, tf.int32)
            return real_length

        batch_length = get_length(non_zero_weights)

        self.initial_state = cell.zero_state(config.batch_size, data_type())
        state = self.initial_state

        cell = tf.contrib.rnn.LSTMBlockCell(size, forget_bias = 1)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
        cell = tf.contrib.rnn.MultiRNNCell(cells=[cell]*self.num_layers, state_is_tuple=True)

        self.cell = cell


        # output embedding
        self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                [vocab_size, size], dtype=tf.float32)

        self.output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                [vocab_size], dtype=tf.float32)

        output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
            sequence_length=batch_length, initial_state=state, dtype=tf.float32)


        def output_embedding(current_output):
            return tf.add(tf.matmul(current_output, tf.transpose(self.output_embedding_mat)),
                            self.output_embedding_bias)

        # Compute logits

        logits = tf.map_fn(output_embedding, output)
        logits = tf.reshape(logits, [-1, vocab_size])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
            tf.reshape(self.output_batch, [-1]), logits = logits) \
            * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)



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

    def batch_train(self, sess, saver, config):
        """Runs the model on the given data."""
        best_score = np.inf
        patience = 5
        epoch = 0

        for i in range(self.max_max_epoch):

            lr_decay = self.lr_decay ** max(i + 1 - self.max_epoch, 0.0)
            self.learning_rate = self.assign_lr(config.learning_rate * lr_decay)
            sess.run(self.training_init_op, {self.file_name_train: "./data/ptb.train.txt.ids"})
            state = sess.run(model.initial_state)
            costs = 0.0
            train_valid_words = 0
            iters = 0
            for step in range(self.train_epoch):
                start_time = time.time()

                cost, _valid_words, current_learning_rate, final_state, _ = sess.run(
                    [self.cost, self.valid_words, self.learning_rate, model.final_state self.updates],
                    {self.dropout_rate:0.5})
                costs += cost

                iters += self.num_steps
                if step % (self.train_epoch // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step * 1.0 /self.train_epoch, np.exp(costs/iters),
                        iters * self.batch_size / (time.time() - start_time)))
                train_valid_words = 0

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(self.learning_rate)))
            sess.run(self.validation_init_op, {self.file_name_validation: "./data/ptb.valid.txt.ids"})
            dev_costs = 0.0
            dev_valid_words = 0
            state = sess.run(model.initial_state)
            iters = 0
            for step in range(self.valid_epoch):
                start_time = time.time()
                dev_cost, _dev_valid_words, final_state = sess.run(
                    [self.cost, self.valid_words, model.final_state], {self.dropout_rate: 1.0})

                dev_costs += dev_cost
                dev_valid_words += _dev_valid_words
                iters += self.num_steps
                if step % (self.valid_epoch // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step % 1.0/self.valid_epoch, np.exp(dev_costs/iters),
                        iters * self.batch_size /(time.time() - start_time)))

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(self.learning_rate)))
            saver.save(sess, "model/best_model.ckpt")

    def predict(self, sess, input_file, raw_file, verbose=False):
        # if verbose is trrue, then we print the ppl of every sequence

        sess.run(self.test_init_op, {self.file_name_test: input_file})

        with open(raw_file) as fp:

            global_dev_loss = 0.0
            global_dev_valid_words = 0

            for raw_line in fp.readlines():

                raw_line = raw_line.strip()

                _dev_loss, _dev_valid_words, input_line = ses.run(
                    [self.loss, self.valid_words, self.input_batch],
                    {self.dropout_rate: 1.0})

                dev_loss = np.sum(_dev_loss)
                dev_valid_words = _dev_valid_words

                global_dev_loss += dev_loss
                global_dev_valid_words += dev_valid_words

                if verbose:
                    dev_loss /= dev_valid_words
                    dev_ppl = math.exp(dev_loss)
                    print(raw_line + " Test PPL: %.3f" % (dev_ppl))

            global_dev_loss /= global_dev_valid_words
            global_dev_ppl = math.exp(global_dev_loss)
            print("Global Test PPL: %.3f" % (global_dev_ppl))
