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

flags = tf.flags
flags.DEFINE_string("vocab_file", "../RNN/data/vocab.txt",
    "File containing the vocabulary")
FLAGS = flags.FLAGS



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
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
            return input_seq, output_seq
        '''
        with tf.gfile.GFile("data/ptb.train.txt.ids") as f:
            lines = f.readlines()
            print(lines)
            train_data = []
            for line in lines:
                train_data.append((parse(line)))
        print(1)
        '''
        training_dataset0 = tf.data.TextLineDataset(self.file_name_train).map(parse).padded_batch(config.batch_size, padded_shapes=([None],[None]))
        # gets next batch so that wen can combine the two datasets into one
        training_dataset1 = training_dataset0.skip(1)
        training_dataset = tf.data.Dataset.zip((training_dataset0, training_dataset1))

        validation_dataset0 = tf.data.TextLineDataset(self.file_name_validation).map(parse).padded_batch(config.batch_size, padded_shapes=([None],[None]))
        validation_dataset1 = validation_dataset0.skip(1)
        validation_dataset = tf.data.Dataset.zip((validation_dataset0, validation_dataset1))

        test_dataset0 = tf.data.TextLineDataset(self.file_name_test).map(parse).batch(1)
        test_dataset1 = test_dataset0.skip(1)
        test_dataset = tf.data.Dataset.zip((test_dataset0, test_dataset1))

        temp_iter = tf.data.Iterator.from_structure(training_dataset0.output_types,
            training_dataset0.output_shapes)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
            training_dataset.output_shapes)




        t_ = temp_iter.make_initializer(training_dataset0)
        t__ = temp_iter.make_initializer(training_dataset1)
        v_ = temp_iter.make_initializer(validation_dataset0)
        v__ = temp_iter.make_initializer(validation_dataset1)
        te_ = temp_iter.make_initializer(test_dataset0)
        te__ = temp_iter.make_initializer(test_dataset1)
        self.training_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)


        batch0, batch1 = iterator.get_next()

        #self.input_batch0 = self.input_batch1 = batch0
        #self.output_batch0 = self.output_batch1 = batch1
        self.input_batch0, self.output_batch0 = batch0
        self.input_batch1, self.output_batch1 = batch1

        # input embedding

        embedding = tf.get_variable(
            "embedding", [self.vocab_size, size], dtype=tf.float32)
        inputs0 = tf.nn.embedding_lookup(embedding, self.input_batch0)
        inputs1 = tf.nn.embedding_lookup(embedding, self.input_batch1)

        non_zero_weights = tf.sign(self.input_batch0)
        non_zero_weights1 = tf.sign(self.input_batch1)
        self.valid_words = tf.reduce_sum(non_zero_weights)

        # Compute sequence length
        def get_length(non_zero_place):
            real_length = tf.reduce_sum(non_zero_place, 1)
            real_length = tf.cast(real_length, tf.int32)
            return real_length

        batch_length = get_length(non_zero_weights)


        cell0 = tf.contrib.rnn.LSTMBlockCell(size, forget_bias = 1, use_peephole=True, reuse=tf.AUTO_REUSE)
        cell0 = tf.contrib.rnn.DropoutWrapper(cell0, input_keep_prob=self.keep_prob)

        cell1 = tf.contrib.rnn.LSTMBlockCell(size, forget_bias = 1, use_peephole=True, reuse=tf.AUTO_REUSE)
        cell1 = tf.contrib.rnn.DropoutWrapper(cell1, input_keep_prob=self.keep_prob)


        self.initial_state = cell0.zero_state(self.batch_size, tf.float32)
        state = self.initial_state

        self.cell0 = cell0
        self.cell1 = cell1

        # output embedding
        self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                [vocab_size, size], dtype=tf.float32)

        self.output_embedding_bias = tf.get_variable("output_embedding_bias",
                                                [vocab_size], dtype=tf.float32)

        output0, state0 = tf.nn.dynamic_rnn(cell=cell0, inputs = inputs0,
            sequence_length=batch_length, initial_state=state, dtype=tf.float32)

        output1, state = tf.nn.dynamic_rnn(cell=cell1, inputs = inputs1,
            sequence_length=batch_length, initial_state=state0, dtype=tf.float32)

        def output_embedding(current_output):
            return tf.add(tf.matmul(current_output, tf.transpose(self.output_embedding_mat)),
                            self.output_embedding_bias)

        # Compute logits

        logits = tf.map_fn(output_embedding, output1)
        logits = tf.reshape(logits, [-1, vocab_size])
        self.probas = tf.nn.softmax(logits, name='p')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=
            tf.reshape(self.output_batch1, [-1]), logits = logits) \
            * tf.cast(tf.reshape(non_zero_weights1, [-1]), tf.float32)



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
            sess.run(self.training_init_op, {self.file_name_train: "./data/" + train_file + ".ids"})
            state = sess.run(self.initial_state)
            costs = 0.0
            iters = 0
            for step in range(self.train_epoch):
                start_time = time.time()

                cost, current_learning_rate, final_state, _ = sess.run(
                    [self.cost, self.learning_rate, self.final_state, self.updates])
                costs += cost

                iters += self.num_steps
                if step % (self.train_epoch // 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step * 1.0 /self.train_epoch, np.exp(costs/iters),
                        iters * self.batch_size / (time.time() - start_time)))

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(self.learning_rate)))
            sess.run(self.validation_init_op, {self.file_name_validation: "./data/" + valid_file + ".ids"})
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
            saver.save(sess, "model/best_model.ckpt")

    def predict(self, sess, input_file, raw_file, verbose=False):
        # if verbose is trrue, then we print the ppl of every sequence

        sess.run(self.test_init_op, {self.file_name_test: input_file})

        with open(raw_file) as fp:

            global_dev_loss = 0.0

            for raw_line in fp.readlines():

                raw_line = raw_line.strip()

                dev_cost, input_line0, input_line1 = sess.run(
                    [self.cost, self.input_batch0, self.input_batch1])



                dev_costs += dev_cost

                if verbose:
                    dev_ppl = np.exp(dev_cost)
                    print(raw_line + " Test PPL: %.3f" % (dev_ppl))

            dev_ppl = math.exp(dev_costs)
            print("Global Test PPL: %.3f" % (dev_ppl))
