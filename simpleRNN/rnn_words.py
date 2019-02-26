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
        self.training_file = "RNN/data/train.txt"#

        self.training_data = self.read_data(self.training_file)
        print("Loaded training data...")
        self.dictionary, self.reverse_dictionary = self.build_dataset(self.training_data)

        self.embedding_model = gensim.models.Word2Vec.load(self.path_to_model + "my_embedding_model")

        self.vocab_size = len(self.embedding_model.wv.vocab)
        self.weights = {'out': self.embedding_model.syn1neg}
        # tf Graph input
        self.x = tf.placeholder("float", [self.batch_size, self.n_input, self.n_hidden])
        self.y = tf.placeholder("float", [self.batch_size, self.vocab_size])

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

    def set_model_name(self, n):
        self.model_name = n

    def set_path_to_model(self, n):
        self.path_to_model = n

    def elapsed(self,sec):
        if sec<60:
            return str(sec) + " sec"
        elif sec<(60*60):
            return str(sec/60) + " min"
        else:
            return str(sec/(60*60)) + " hr"

    def read_data(self,fname):
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [word for i in range(len(content)) for word in content[i].split()]
        content = np.array(content)
        return content

    def build_dataset(self, words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary
    #vocab_size = len(dictionary)

    def RNN(self):
        # 2-layer LSTM, each layer has self.n_hidden units.
        # Average Accuracy= 95.20% at 50k iter
        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])

        # 1-layer LSTM with self.n_hidden units but with lower accuracy.
        # Average Accuracy= 90.60% 50k iter
        # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
        # rnn_cell = rnn.BasicLSTMCell(self.n_hidden)

        # generate prediction
        # the shape of outputs is [self.batch_size, self.n_input, self.n_hidden]
        outputs, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs = self.x, dtype = tf.float32)
        output = states[-1].h
        #outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are self.n_input outputs but
        # we only want the last output
        return tf.matmul(output, tf.transpose(self.weights['out']))

    def train(self):
        with tf.Session() as session:
            session.run(self.init)
            step = 0
            offset = random.randint(0,self.n_input+1)
            end_offset = self.n_input + 1
            acc_total = 0
            loss_total = 0

            self.writer.add_graph(session.graph)

            while step < self.training_iters:
                # Generate a minibatch. Add some randomness on selection process.
                if offset > (len(self.training_data)-end_offset):
                    offset = random.randint(0, self.n_input+1)
                symbols = []
                for i in range(self.batch_size):
                    symbol = [str(self.training_data[j]) for j in range(offset+i, offset+self.n_input+i)]
                    symbols.append(symbol)
                embedded_batch = []
                for batch in symbols:
                    embedded_symbols = []
                    for word in batch:
                        try:
                            embedding = self.embedding_model.wv[word]
                        except KeyError:
                            print(word + " not in vocabulary")
                            embedding = np.zeros((300,), dtype=np.float)
                        embedded_symbols.append(embedding)
                    embedded_batch.append(embedded_symbols)

                # embeded_symbols shape [self.batch_size, self.n_input, self.n_hidden]

                symbols_out_onehot = np.zeros([self.batch_size, self.vocab_size], dtype=float)
                for i in range(self.batch_size):
                    try:
                        symbols_out_onehot[i][self.embedding_model.wv.vocab[self.training_data[offset+self.n_input+i]].index] = 1.0
                    except:
                        symbols_out_onehot[i][0] = 1.0
                #print(embedded_batch)
                symbols_out_onehot = np.reshape(symbols_out_onehot,[self.batch_size,-1])


                _, acc, loss, embedding_pred = session.run([self.optimizer, self.accuracy, self.cost, self.probas], \
                                                        feed_dict={self.x: embedded_batch, self.y: symbols_out_onehot})
                predictions = []
                for prediction in embedding_pred:
                    predictions.append(self.embedding_model.wv.index2word[prediction])


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
                    #symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                    for batch in range(self.batch_size):
                        print("%s - [%s] vs [%s]" % (symbols_in[batch],symbols_out[batch],predictions[batch]))
                step += 1
                offset += (self.n_input+1)
            print("Optimization Finished!")
            print("Elapsed time: ", self.elapsed(time.time() - self.start_time))
            print("Run on command line.")
            print("\ttensorboard --logdir=%s" % (self.logs_path))
            print("Point your web browser to: http://localhost:6006/")
            saver = tf.train.Saver()# -*- coding: utf-8 -*-

            saver.save(session, self.path_to_model+"/"+self.model_name)

    # generate a suggestion given a session and a string _input
    def generate_suggestion(self, session, _input):
        input_sent = word_tokenize(_input)
        embedded_symbols = []
        if len(input_sent) == self.n_input:
            try:
                for word in input_sent:
                    print(word)
                    try:
                        embedding = self.embedding_model.wv[word]
                    except KeyError:
                        print(word + " not in vocabulary")
                        embedding = np.zeros((300,), dtype=np.float)
                    embedded_symbols.append(embedding)
                # embeded_symbols shape [1, n_input, n_hidden]
                embedded_symbols = [embedded_symbols]
                output_sent = ""
                for i in range(23):
                    onehot_pred = session.run(self.probas, feed_dict={self.x: embedded_symbols})
                    onehot_pred = self.embedding_model.wv.index2word[onehot_pred[0]]
                    output_sent += " %s" % (onehot_pred)
                    embedded_symbols = embedded_symbols[0][1:]
                    embedded_symbols.append(self.embedding_model.wv[onehot_pred])
                    embedded_symbols = [embedded_symbols]
                return output_sent
            except Exception as e:
                print(e)
                return e

    def run(self):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(self.path_to_model))
            while True:
                prompt = "%s words: " % self.n_input
                input_sent = input(prompt)
                input_sent = word_tokenize(input_sent)
                embedded_symbols = []
                if len(input_sent) != self.n_input:
                    continue
                try:
                    for word in input_sent:
                        try:
                            embedding = self.embedding_model.wv[word]
                        except KeyError:
                            print(word + " not in vocabulary")
                            embedding = np.zeros((300,), dtype=np.float)
                        embedded_symbols.append(embedding)
                    # embeded_symbols shape [1, n_input, n_hidden]
                    embedded_symbols = [embedded_symbols]
                    output_sent = "%s" % (input_sent)
                    for i in range(23):
                        onehot_pred = session.run(self.probas, feed_dict={self.x: embedded_symbols})
                        onehot_pred = self.embedding_model.wv.index2word[onehot_pred[0]]
                        output_sent +=  " %s" % (onehot_pred)
                        embedded_symbols = embedded_symbols[0][1:]
                        embedded_symbols.append(self.embedding_model.wv[onehot_pred])
                        embedded_symbols = [embedded_symbols]
                    print(output_sent)
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    args = sys.argv[1:]
    d = {"learning_rate": 0.001, "training_iters" : 1000,"n_input" : 4,"batch_size" : 2, "n_hidden" : 300}
    path_to_model = "RNN/models/"
    model_name = "best_model"
    display_step = 100
    if len(args) >= 1 and args[0] == "train":
        rnn = SimpleRNN(d, display_step, path_to_model, model_name)
        rnn.train()
    elif (len(args) >= 1 and args[0] == "manual"):
        d["batch_size"] = 1
        rnn = SimpleRNN(d, display_step, path_to_model, model_name)
        #rnn.train()
        sess = rnn.load()
        s = input("enter a sent(4): ")
        while s != "quit":
            out_ = rnn.generate_suggestion(sess, s)
            s = input("enter a sent(4): ")
        rnn.close(sess)
    else:
        d["batch_size"] = 1
        rnn = SimpleRNN(d, display_step, path_to_model, model_name)
        rnn.run()
