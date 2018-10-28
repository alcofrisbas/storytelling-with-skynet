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



class RNNModel(object):
    """model"""
    def __init__(self, vocab_size, config, num_train_samples, num_valid_samples,
        num_test_samples):
        self.vocab_size = vocab_size
        self.batch_size = config.batch_size
        self.max_epoch = config.max_epoch
        self.max_max_epoch = config.max_max_epoch
        self.num_train_samples = num_train_samples
        self.checkpoint_step = 100
        self.num_valid_samples = num_valid_samples
        self.num_test_samples = num_test_samples
        self._rnn_params = None
        self._cell = None
        self.num_layers = config.num_layers
        self.num_steps = config.num_steps
        self.max_gradient_norm=5.0
        size = config.hidden_size

        self.global_step = tf.Variable(0, trainable=False)
        # We set a dynamic learining rate, it decays every time the model has gone through 150 batches.
        # A minimum learning rate has also been set.
        self.learning_rate = tf.train.exponential_decay(config.learning_rate, self.global_step,
                                           150, 0.96, staircase=True)
        self.learning_rate = tf.cond(tf.less(self.learning_rate, 0.001), lambda: tf.constant(0.001),
            lambda: self.learning_rate)

        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.file_name_train = tf.placeholder(tf.string)
        self.file_name_validation = tf.placeholder(tf.string)
        self.file_name_test = tf.placeholder(tf.string)



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

        cell = tf.contrib.rnn.LSTMBlockCell(size, forget_bias = 1)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
        cell = tf.contrib.rnn.MultiRNNCell(cells=[cell]*self.num_layers, state_is_tuple=True)

        self.cell = cell


        # output embedding
        output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                [vocab_size, size], dtype=tf.float32)

        output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                [vocab_size], dtype=tf.float32)

        output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
            sequence_length=batch_length,dtype=tf.float32)


        def output_embedding(current_output):
            return tf.add(tf.matmul(current_output, tf.transpose(output_embedding_mat)),
                            output_embedding_bias)

        # Compute logits

        logits = tf.map_fn(output_embedding, output)
        logits = tf.reshape(logits, [-1, vocab_size])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
            tf.reshape(self.output_batch, [-1]), logits = logits) \
            * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)

        '''
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b =tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        self.probas = tf.nn.softmax(logits, name='probas')
        #target = tf.reshape(target,(self.batch_size, self.num_steps))
        # Use the contrib sequence loss and average over the batches

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            target,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        '''

        self.loss = loss

        # Train

        params = tf.trainable_variables()

        #optimizer = tf.train.GradientDescentOptimizer(self._lr)
        opt= tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10e-4)
        gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def batch_train(self, sess, saver):
        """Runs the model on the given data."""
        best_score = np.inf
        patience = 5
        epoch = 0

        while epoch < self.max_max_epoch:
            sess.run(self.training_init_op, {self.file_name_train: "./data/ptb.train.txt.ids"})
            train_loss = 0.0
            train_valid_words = 0
            while True:

                try:
                    _loss, _valid_words, global_step, current_learning_rate, _ = sess.run(
                        [self.loss, self.valid_words, self.global_step, self.learning_rate, self.updates],
                        {self.dropout_rate:0.5})
                    train_loss += np.sum(_loss)
                    train_valid_words += _valid_words

                    if global_step % self.checkpoint_step == 0:

                        train_loss /= train_valid_words
                        train_ppl = math.exp(train_loss)
                        print("Training Step: %d, LR: %d" % (global_step, current_learning_rate))
                        print("Training PPL: %d" % (train_ppl))

                        train_loss = 0.0
                        train_valid_words = 0

                except tf.errors.OutOfRangeError:
                    # The end of one epoch
                    break

            sess.run(self.validation_init_op, {self.file_name_validation: "./data/ptb.valid.txt.ids"})
            dev_loss = 0.0
            dev_valid_words = 0
            while True:
                try:
                    _dev_loss, _dev_valid_words = sess.run(
                        [self.loss, self.valid_words], {self.dropout_rate: 1.0})

                    dev_loss += np.sum(_dev_loss)
                    dev_valid_words += _dev_valid_words

                except tf.error.OutOfRangeError:
                    dev_loss /= dev_valid_words
                    dev_ppl = math.exp(dev_loss)
                    print("Validation PPL: %d" % (dev_ppl))
                    if dev_ppl < bets_score:
                        patience = 5
                        saver.save(sess, "model/best_model.ckpt" )
                        best_score = dev_ppl
                    else:
                        patience -= 1

                    if patience == 0:
                        epoch = self.max_max_epochs

                    break


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
                        print(raw_line + " Test PPL: %d" % (dev_ppl))

                global_dev_loss /= global_dev_valid_words
                global_dev_ppl = math.exp(global_dev_loss)
                print("Global Test PPL: %d" % (global_dev_ppl))
