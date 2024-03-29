'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function
import random
import collections
import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from nltk.tokenize import word_tokenize
import gensim
from numpy import genfromtxt


"""
learning_rate = 0.001
training_iters = 50000
n_input = 4
batch_size = 2
n_hidden = 300
"""


class SimpleRNN:
    # Parameters
    def __init__(self, d, display_step, path_to_model, model_name):
        self.__dict__.update(d)
        self.display_step = display_step
        self.path_to_model = path_to_model
        self.model_name = model_name

        self.start_time = time.time()
        # Target log path
        self.logs_path = '/tmp/tensorflow/rnn_word'
        self.writer = tf.summary.FileWriter(self.logs_path)

        # Text file containing words for training
        self.training_file = d["training_file"]
        # training .txt file
        self.training_data = self.read_data(self.training_file)
        print("Loaded training data...")
        # pull embeddings from models
        self.embedding_model = gensim.models.Word2Vec.load(self.path_to_model + self.model_name)
        self.input_embedding_matrix = np.load(self.path_to_model + self.model_name + "_input_embedding_model.npy")
        self.output_embedding_matrix = tf.cast(np.load(self.path_to_model + self.model_name + "_output_embedding_model.npy"), tf.float32)

        self.vocab_size = self.input_embedding_matrix.shape[0]
        self.weights = {'out': self.output_embedding_matrix}
        # tf Graph input
        self.x = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.vocab_size])
        # model predictions
        self.pred = self.RNN()
        self.probas = tf.argmax(self.pred, 1)
        # Loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Model evaluation
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initializing the variables
        self.init = tf.global_variables_initializer()

        self.index2word = genfromtxt(self.path_to_model + self.model_name + "_vocab.csv",  dtype=str)

    def set_model_name(self, n):
        self.model_name = n

    def set_path_to_model(self, n):
        self.path_to_model = n
    # simple function that gets the time elapsed
    def elapsed(self,sec):
        if sec<60:
            return str(sec) + " sec"
        elif sec<(60*60):
            return str(sec/60) + " min"
        else:
            return str(sec/(60*60)) + " hr"
    """
    Reads data from a .txt file and returns a list of words as they appear in the file
    input: A file to be read
    output: A list of strings
    """
    def read_data(self,fname):
        sentences = []
        with open(fname) as f:
            content = f.readlines()
        for sent in content:
            sent = word_tokenize(sent)
            for word in sent:
                sentences.append(word)
        return sentences

    # core rnn calculations
    def RNN(self):
        # embed the input to size [self.batch_size, self.n_input, self.n_hidden]
        embedded_input = tf.cast(tf.nn.embedding_lookup(self.input_embedding_matrix, self.x), tf.float32)
        # 2-layer LSTM, each layer has self.n_hidden units.
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])

        # generate prediction
        # the shape of outputs is [self.batch_size, self.n_input, self.n_hidden]
        outputs, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs = embedded_input, dtype = tf.float32)
        # there are self.n_input outputs but
        # we only want the last output
        output = states[-1].h

        # calculate wx+b to get prediction of shape [self.batch_size, self.vocab_size]
        return tf.matmul(output, tf.transpose(self.weights['out']))

    # a function that trains the model
    def train(self):
        with tf.Session() as session:
            # initialize tf session
            session.run(self.init)
            step = 0
            offset = random.randint(0,self.n_input+1)
            end_offset = self.n_input + 1
            acc_total = 0
            loss_total = 0
            self.writer.add_graph(session.graph)

            # will train for self.training_iters steps
            while step < self.training_iters:
                # Generate a minibatch. Add some randomness on selection process.
                if offset > (len(self.training_data)-end_offset):
                    offset = random.randint(0, self.n_input+1)
                symbols = []
                # get a subset of words to read
                for i in range(self.batch_size):
                    symbol = [str(self.training_data[j]) for j in range(offset+i, offset+self.n_input+i)]
                    symbols.append(symbol)
                # construct the input of shape [self.batch_size, self.n_input]
                embedded_batch = []
                for batch in symbols:
                    embedded_symbols = []
                    for word in batch:
                        # convert each word from string to one-hot vector
                        try:
                            embedding = self.embedding_model.wv.vocab[word.lower()].index
                        except KeyError:
                            #print(word + " not in vocabulary")
                            embedding = len(self.input_embedding_matrix)-2
                        embedded_symbols.append(embedding)
                    embedded_batch.append(embedded_symbols)

                # construct word to be targeted
                symbols_out_onehot = np.zeros([self.batch_size, self.vocab_size], dtype=float)
                for i in range(self.batch_size):
                    try:
                        symbols_out_onehot[i][self.embedding_model.wv.vocab[self.training_data[offset+self.n_input+i]].index] = 1.0
                    except:
                        symbols_out_onehot[i][0] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[self.batch_size,-1])

                # get prediction, accuracy, and loss
                _, acc, loss, embedding_pred = session.run([self.optimizer, self.accuracy, self.cost, self.probas], \
                                                        feed_dict={self.x: embedded_batch, self.y: symbols_out_onehot})
                # change integer predictions to strings
                predictions = []
                for prediction in embedding_pred:
                    predictions.append(self.index2word[prediction])

                # update loss and accuracy
                loss_total += loss
                acc_total += acc
                if (step+1) % self.display_step == 0:
                    print("Iter= " + str(step+1) + ", Average Loss= " + \
                          "{:.6f}".format(loss_total/self.display_step) + ", Average Accuracy= " + \
                          "{:.2f}%".format(100*acc_total/self.display_step))
                    print("Elapsed time: ", self.elapsed(time.time() - self.start_time))
                    acc_total = 0
                    loss_total = 0
                    symbols_in = []
                    symbols_out = []
                    for batch in range(self.batch_size):
                        symbol_in = [self.training_data[i] for i in range(offset+batch, offset + self.n_input + batch)]
                        symbols_in.append(symbol_in)
                        symbol_out = self.training_data[offset + self.n_input+batch]
                        symbols_out.append(symbol_out)
                    for batch in range(self.batch_size):
                        print("%s - [%s] vs [%s]" % (symbols_in[batch],symbols_out[batch],predictions[batch]))


                step += 1
                offset += (self.n_input+1)
                # save a new model every 5000 iters
                if step % 1000 == 0:
                    saver = tf.train.Saver()# -*- coding: utf-8 -*-

                    saver.save(session, self.path_to_model+"/"+self.model_name)

            print("Optimization Finished!")
            print("Elapsed time: ", self.elapsed(time.time() - self.start_time))
            print("Run on command line.")
            print("\ttensorboard --logdir=%s" % (self.logs_path))
            print("Point your web browser to: http://localhost:6006/")

    # generate a suggestion given a session and a string sentence
    # def generate_suggestion(self, session, sentence):
    #     sentence = sentence.strip()
    #     input_sent = word_tokenize(sentence)
    #     embedded_symbols = []
    #     try:
    #         for word in input_sent:
    #             try:
    #                 embedding = self.embedding_model.wv.vocab[word.lower()].index
    #             except KeyError:
    #                 print(word + " not in vocabulary")
    #                 embedding = len(self.input_embedding_matrix)-2
    #             embedded_symbols.append(embedding)
    #         # embeded_symbols shape [1, n_input, n_hidden]
    #         embedded_symbols = [embedded_symbols]
    #         onehot_pred = 0
    #         output_sent = ""
    #         length = 0
    #         while onehot_pred != "." and onehot_pred != "!" and onehot_pred != "?" and length < 23:
    #             length += 1
    #             onehot_pred = session.run(self.pred, feed_dict={self.x: embedded_symbols})
    #             # if np.argmax(onehot_pred, 1)[0] == len(self.input_embedding_matrix)-2 or np.argmax(onehot_pred, 1)[0] == 0:
    #             #     largest = 0
    #             #     largest_index = 0
    #             #     for i in range(len(onehot_pred[0])):
    #             #         if onehot_pred[0][i] > largest:
    #             #             largest = onehot_pred[0][i]
    #             #             largest_index = i
    #             #     onehot_pred[0][i] = 0
    #             #     second_largest = 0
    #             #     second_largest_index = 0
    #             #     for i in range(len(onehot_pred[0])):
    #             #         if onehot_pred[0][i] > second_largest:
    #             #             second_largest = onehot_pred[0][i]
    #             #             second_largest_index = i
    #             #     onehot_pred = second_largest_index
    #             # else:
    #             #     onehot_pred = np.argmax(onehot_pred, 1)
    #
    #             onehot_pred = self.index2word[onehot_pred[0]]
    #             if onehot_pred == "PAD" or onehot_pred == "UNK" or onehot_pred == "GO":
    #                 continue
    #             elif onehot_pred == "," or onehot_pred == ";" or onehot_pred == ":":
    #                 output_sent += "%s" % (onehot_pred)
    #             else:
    #                 output_sent += " %s" % (onehot_pred)
    #             embedded_symbols = embedded_symbols[0][1:]
    #             embedded_symbols.append(self.embedding_model.wv.vocab[onehot_pred.lower()].index)
    #             embedded_symbols = [embedded_symbols]
    #
    #         if output_sent != "":
    #             output_sent = output_sent.strip().capitalize()
    #             if onehot_pred in (".", "!", "?"):
    #                 output_sent += "%s" % (onehot_pred)
    #             else:
    #                 output_sent += "."
    #         else:
    #             output_sent = "Sorry, I don't have any suggestions for that one."
    #
    #         return output_sent
    #     except Exception as e:
    #         print(e)
    #         return e

    def generate_suggestion(self, session, sentence):
        sentence = sentence.strip()
        input_sent = word_tokenize(sentence)
        embedded_symbols = []
        try:
            for word in input_sent:
                try:
                    embedding = self.embedding_model.wv.vocab[word.lower()].index
                except KeyError:
                    print(word + " not in vocabulary")
                    embedding = len(self.input_embedding_matrix)-2
                embedded_symbols.append(embedding)
            # embeded_symbols shape [1, n_input, n_hidden]
            embedded_symbols = [embedded_symbols]
            onehot_pred = 0
            output_sent = ""
            length = 0
            while onehot_pred != "." and onehot_pred != "!" and onehot_pred != "?" and length < 23:
                length += 1
                onehot_pred = session.run(self.pred, feed_dict={self.x: embedded_symbols})
                if np.argmax(onehot_pred, 1)[0] == len(self.input_embedding_matrix)-2:
                    largest = 0
                    largest_index = 0
                    for i in range(len(onehot_pred[0])):
                        if onehot_pred[0][i] > largest:
                            largest = onehot_pred[0][i]
                            largest_index = i
                    onehot_pred[0][i] = 0
                    second_largest = 0
                    second_largest_index = 0
                    for i in range(len(onehot_pred[0])):
                        if onehot_pred[0][i] > second_largest:
                            second_largest = onehot_pred[0][i]
                            second_largest_index = i
                    onehot_pred = second_largest_index
                else:
                    onehot_pred = np.argmax(onehot_pred, 1)

                onehot_pred = self.index2word[onehot_pred[0]]
                if onehot_pred == "PAD" or onehot_pred == "UNK" or onehot_pred == "GO":
                    continue
                elif onehot_pred == "," or onehot_pred == ";" or onehot_pred == ":":
                    output_sent += "%s" % (onehot_pred)
                else:
                    output_sent += " %s" % (onehot_pred)
                embedded_symbols = embedded_symbols[0][1:]
                embedded_symbols.append(self.embedding_model.wv.vocab[onehot_pred.lower()].index)
                embedded_symbols = [embedded_symbols]

            if output_sent != "":
                output_sent = output_sent.strip().capitalize()
                if onehot_pred in (".", "!", "?"):
                    output_sent += "%s" % (onehot_pred)
                else:
                    output_sent += "."
            else:
                output_sent = "Sorry, I don't have any suggestions for that one."

            return output_sent
        except Exception as e:
            print(e)
            return e

    def run(self):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(self.path_to_model))
            while True:
                prompt = "write a sentence: "
                input_sent = input(prompt)
                input_sent = word_tokenize(input_sent)
                embedded_symbols = []
                try:
                    for word in input_sent:
                        try:
                            embedding = self.embedding_model.wv.vocab[word.lower()].index
                        except KeyError:
                            print(word + " not in vocabulary")
                            embedding = len(self.input_embedding_matrix)-2
                        embedded_symbols.append(embedding)
                    # embeded_symbols shape [1, n_input, n_hidden]
                    output_sent = ""
                    output_word = ""
                    embedded_symbols = [embedded_symbols]
                    for i in range(23):
                        onehot_pred = session.run(self.pred, feed_dict={self.x: embedded_symbols})
                        if tf.argmax(onehot_pred, 1) == len(self.input_embedding_matrix)-2:
                            largest = 0
                            largest_index = 0
                            for i in range(len(onehot_pred[0])):
                                if onehot_pred[0][i] > largest:
                                    largest = onehot_pred[0][i]
                                    largest_index = i
                            onehot_pred[0][i] = 0
                            second_largest = 0
                            second_largest_index = 0
                            for i in range(len(onehot_pred[0])):
                                if onehot_pred[0][i] > second_largest:
                                    second_largest = onehot_pred[0][i]
                                    second_largest_index = i
                            onehot_pred = second_largest_index
                        else:
                            onehot_pred = tf.argmax(onehot_pred, 1)

                        #print(onehot_pred)
                        onehot_pred = self.index2word[onehot_pred[0]]
                        if onehot_pred == "PAD" or onehot_pred == "UNK" or onehot_pred == "GO":
                            continue
                        output_sent +=  " %s" % (onehot_pred)
                        embedded_symbols = embedded_symbols[0][1:]
                        embedded_symbols.append(self.embedding_model.wv.vocab[onehot_pred.lower()].index)
                        embedded_symbols = [embedded_symbols]
                    print(output_sent)
                except Exception as e:
                    print(e)


def run(learning_rate, training_iters, n_input, batch_size, n_hidden, path_to_model, model_name, train, training_file):
    d = {"learning_rate": learning_rate, "training_iters" : training_iters,"n_input" : n_input,"batch_size" : batch_size, "n_hidden" : n_hidden, "training_file": training_file}
    display_step = 1000
    if train:
        rnn = SimpleRNN(d, display_step, path_to_model, model_name)
        rnn.train()
    else:
        d["batch_size"] = 1
        rnn = SimpleRNN(d, display_step, path_to_model, model_name)
        rnn.run()
