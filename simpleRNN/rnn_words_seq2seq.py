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
import csv
from numpy import genfromtxt


"""
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 4
batch_size = 2
# number of units in RNN cell
n_hidden = 300
path_to_model = "RNN/models/"
"""

class SimpleRNN:
    # Parameters
    def __init__(self, learning_rate, training_iters, display_step, n_input,
    batch_size, n_hidden, path_to_model, model_name, to_train):
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.display_step = display_step
        self.n_input = n_input
        self.to_train = to_train
        self.batch_size = batch_size
        # number of units in RNN cell
        self.n_hidden = n_hidden
        self.path_to_model = path_to_model
        self.model_name = model_name

        self.start_time = time.time()
        # Target log path
        self.logs_path = '/tmp/tensorflow/rnn_word'
        self.writer = tf.summary.FileWriter(self.logs_path)

        # Text file containing words for training
        self.training_file = "simpleRNN/data/test.txt"   #'simpleRNN/belling_the_cat.txt'
        self.training_data = self.read_data(self.training_file)
        self.output_seq_length = len(self.training_data[1]) - 1
        print("Loaded training data...")
    #    self.dictionary, self.reverse_dictionary = self.build_dataset(self.training_data)

        self.embedding_model = gensim.models.Word2Vec.load(self.path_to_model + "basic_model")

        #file containing the index to vector including the paddings
        self.input_embedding_matrix = np.load(self.path_to_model + "basic_model_input_embedding_model.npy")
        self.output_embedding_matrix = tf.cast(np.load(self.path_to_model + "basic_model_output_embedding_model.npy"), tf.float32)
        self.vocab_size = self.input_embedding_matrix.shape[0]
        self.weights = {'out': self.output_embedding_matrix}
        # tf Graph input
        self.padded_lengths = 9
        self.x = tf.placeholder(tf.int32, [None, self.padded_lengths])
        self.y = tf.placeholder(tf.int32, [None, self.padded_lengths])
        self.encoder_lengths = tf.placeholder(tf.int32, shape=(None,), name="encoder_length")
        self.decoder_lengths = tf.placeholder(tf.int32, shape=(None,), name="decoder_length")
        self.start_token = len(self.input_embedding_matrix) - 3 # GO
        self.end_token = len(self.input_embedding_matrix) - 1  # PAD

        self.logits, self.batch_loss, self.valid_predictions  = self.RNN()
        try:
            self.probas = tf.argmax(self.logits, 2)
        except:
            self.probas = tf.argmax(self.logits, 1)
        # Loss and optimizer
        with tf.name_scope("optimization"):
            # optimizer
            try:
                self.cost = tf.reduce_mean(self.batch_loss)
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.batch_loss)
                self.accuracy = tf.reduce_mean(tf.cast(self.valid_predictions, tf.float32))
            except:
                self.cost = 0
                self.optimizer = 0



        # Initializing the variables
        self.init = tf.global_variables_initializer()

        #file containing the vocab
        self.index2word = genfromtxt(self.path_to_model + "basic_model_vocab.csv",  dtype=str)

    def elapsed(self,sec):
        if sec<60:
            return str(sec) + " sec"
        elif sec<(60*60):
            return str(sec/60) + " min"
        else:
            return str(sec/(60*60)) + " hr"

    def read_data(self,fname):
        sentences = []
        with open(fname) as f:
            content = f.readlines()
        for sent in content:
            sent = word_tokenize(sent)
            sentences.append(sent)
        return sentences

    def RNN(self):
        # 2-layer LSTM, each layer has self.n_hidden units.
        # Average Accuracy= 95.20% at 50k iter

        # defining tensors to be fed into graph
        inputs = self.x
        targets = self.y
        # seq2seq embedding layers
        embedded_input = tf.cast(tf.nn.embedding_lookup(self.input_embedding_matrix, inputs), tf.float32)


        # consider using GRU cells
        with tf.variable_scope("encoding") as encoding_scope:
            lstm_encoding = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])
            encoder_outputs, last_state = tf.nn.dynamic_rnn(lstm_encoding, inputs=embedded_input,
                sequence_length=self.encoder_lengths, dtype=tf.float32)

        with tf.variable_scope("decoding") as decoding_scope:
            self.batch_size = tf.shape(inputs)[0]
            lstm_decoding = rnn.BasicLSTMCell(self.n_hidden)#rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden), rnn.BasicLSTMCell(self.n_hidden)])
            # Attention mechanism and wrapper
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.n_hidden, memory=encoder_outputs,
                memory_sequence_length=self.encoder_lengths)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_decoding, attention_mechanism, attention_layer_size=self.n_hidden)
            initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)
            output_layer = tf.layers.Dense(len(self.input_embedding_matrix), name="output_projection")
            if self.to_train:
                embedded_output = tf.cast(tf.nn.embedding_lookup(self.input_embedding_matrix, targets), tf.float32)
                max_dec_length = tf.reduce_max(self.decoder_lengths+1, name="max_dec_length")
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=embedded_output, sequence_length=self.decoder_lengths,
                    name="training_helper")
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state, output_layer=output_layer)

                final_outputs, _final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False, impute_finished=True,
                    maximum_iterations=max_dec_length)
                logits = tf.identity(final_outputs.rnn_output, name="logits")

                targets = tf.slice(self.y, [0,0], [-1, max_dec_length-1], 'targets')

                masks = tf.sequence_mask(self.decoder_lengths+1, max_dec_length-1, dtype=tf.float32, name="masks")
                batch_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=masks,
                    name="batch_loss")

                valid_predictions = tf.identity(final_outputs.sample_id, name="valid_preds")

            else:
                start_tokens = tf.tile(tf.constant([self.start_token], dtype=tf.int32), [self.batch_size],
                    name = "start_tokens")
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.output_embedding_matrix,
                    start_tokens=start_tokens, end_token=self.end_token)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                    helper=inference_helper, initial_state=initial_state, output_layer=output_layer)

                final_outputs, _final_state,_ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder, output_time_major=False, impute_finished=True,
                    maximum_iterations=self.padded_lengths)

                logits = tf.identity(final_outputs.rnn_output, name="predictions")
                batch_loss = None
                valid_predictions = tf.identity(final_outputs.sample_id, name="valid_preds")

            return logits, batch_loss, valid_predictions
        """
            #dec_outputs, _ = tf.nn.dynamic_rnn(lstm_decoding, inputs=embedded_output, initial_state=last_state)
        #logits = tf.matmul(dec_outputs, tf.transpose(self.weights['out']))
        def wxplusb(output):
            return tf.matmul(output, tf.transpose(self.weights['out']))
        self.logits = tf.map_fn(wxplusb, final_outputs.rnn_output)
        #logits = tf.contrib.layers.fully_connected(self.dec_outputs, num_outputs=self.vocab_size,
        #    activation_fn=None)
        return self.logits

        # connect outputs to
        logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=self.vocab_size,
            activation_fn=None)

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
        """
    """
    def generateText(input):
        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(self.path_to_model))
            sentence = input.strip()
            words = sentence.split(' ')
            if len(words) == self.n_input:
                try:
                    symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                    for i in range(32):
                        keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])
                        onehot_pred = session.run(self.pred, feed_dict={x: keys})
                        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                        sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
                        symbols_in_keys = symbols_in_keys[1:]
                        symbols_in_keys.append(onehot_pred_index)
                    print(sentence)
                except:
                    print("Word not in dictionary")
    """
    def train(self):
        with tf.Session() as session:
            session.run(self.init)
            step = 0
            sent_num = 0
            #end_offset = self.n_input + 1
            acc_total = 0
            loss_total = 0

            self.writer.add_graph(session.graph)
            max_size = 0
            for sent in self.training_data:
                if max_size < len(sent):
                    max_size = len(sent)
            while step < self.training_iters:
                # Generate a minibatch. Add some randomness on selection process.
                """
                if offset > (len(self.training_data)-end_offset):
                    offset = random.randint(0, self.n_input+1)
                """
                if sent_num >= len(self.training_data):
                    sent_num = 0
                symbols = self.training_data[sent_num]
                """
                with open("RNN/data/train.txt") as file:

                for i in range(self.batch_size):
                symbol = [str(self.training_data[j]) for j in range(offset+i, offset+self.n_input+i)]
                symbols.append(symbol)
                """
                onehot_batch = []
                """
                for batch in symbols:
                    embedded_symbols = []
                """
                for word in symbols:
                    try:
                        one_hot = self.embedding_model.wv.vocab[word.lower()].index
                        #itemindex = np.where(self.index2word== word)
                    except KeyError:
                        one_hot = len(self.input_embedding_matrix)-2
                    onehot_batch.append(one_hot)
                len_inputs = len(onehot_batch)
                # padding
                while len(onehot_batch) < max_size+1:
                    onehot_batch.append(len(self.input_embedding_matrix)-1)
                onehot_batch = [onehot_batch]
                targets = [len(self.input_embedding_matrix)-3]
                for word in self.training_data[sent_num+1]:
                    try:
                        targets.append(self.embedding_model.wv.vocab[word.lower()].index)
                    except KeyError:
                        targets.append(len(self.input_embedding_matrix)-2)
                len_targets = len(targets)-1
                # padding
                while (len(targets) < max_size+1):
                    targets.append(len(self.input_embedding_matrix)-1)
                targets = [targets]

                """

                for i in range(self.batch_size):
                    try:
                        symbols_out_onehot[i] = self.embedding_model.wv.vocab[self.training_data[offset+self.n_input+i]].index
                    except:
                        symbols_out_onehot[i] = 0
                #print(embedded_batch)

                outputs = []
                for word in self.training_data[sent_num+1][:-1]:
                    #itemindex = np.where(self.index2word== word)
                    #ind = itemindex[0][0]
                    #outputs.append(ind)
                    outputs.append(self.embedding_model.wv.vocab[word].index)
                # padding
                while (len(outputs) < max_size):

                    outputs.append(len(self.input_embedding_matrix)-2)
                outputs = [outputs]
                """
                #symbols_out_onehot = np.reshape(symbols_out_onehot,[self.batch_size,-1])

                #outputs = np.zeros([self.batch_size, self.n_input], dtype=int)
                _, acc, loss, embedding_pred = session.run([self.optimizer, self.accuracy, self.cost, self.probas], \
                                                        feed_dict={self.x: onehot_batch, self.y: targets,
                                                            self.decoder_lengths: [len_targets], self.encoder_lengths: [len_inputs]})
                predictions = []
                for prediction in embedding_pred[0]:
                    predictions.append(self.index2word[prediction])
                    #predictions.append(self.embedding_model.wv.index2word[prediction])
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
                    for batch in range(1):
                        symbol_in = self.training_data[sent_num]#[self.training_data[i] for i in range(offset+batch, offset + self.n_input + batch)]
                        symbols_in.append(symbol_in)
                        symbol_out = self.training_data[sent_num+1]#self.training_data[offset + self.n_input+batch]
                        symbols_out.append(symbol_out)
                    #symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                    for batch in range(1):
                        print("%s - [%s] vs [%s]" % (symbols_in[batch],symbols_out[batch],predictions))
                step += 1
                sent_num += 2
            print("Optimization Finished!")
            print("Elapsed time: ", self.elapsed(time.time() - self.start_time))
            print("Run on command line.")
            print("\ttensorboard --logdir=%s" % (self.logs_path))
            print("Point your web browser to: http://localhost:6006/")
            saver = tf.train.Saver()# -*- coding: utf-8 -*-

            saver.save(session, self.path_to_model+"/"+self.model_name)

    def run(self):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(self.path_to_model))
            while True:
                prompt = "Write a sentence: "
                input_sent = input(prompt)
                input_sent = word_tokenize(input_sent)
                len_inputs = len(input_sent)
                embedded_symbols = []
                try:
                    """
                    for word in input_sent:
                        try:
                            embedding = self.embedding_model.wv[word]
                        except KeyError:
                            print(word + " not in vocabulary")
                            embedding = np.zeros((300,), dtype=np.float)
                        embedded_symbols.append(embedding)
                    # embeded_symbols shape [1, n_input, n_hidden]
                    embedded_symbols = [embedded_symbols]
                    """
                    output_sent = "%s" % (input_sent)
                    input_sent_one_hot = []
                    for word in input_sent:
                        try:
                            input_sent_one_hot.append(self.embedding_model.wv.vocab[word.lower()].index)
                        except KeyError:
                            input_sent_one_hot.append(len(self.input_embedding_matrix)-2)
                    while len(input_sent_one_hot) < self.padded_lengths:
                        input_sent_one_hot.append(len(self.input_embedding_matrix)-3)
                    input_sent = [input_sent_one_hot]
                   # for i in range(23):
                    onehot_pred = session.run(self.probas, feed_dict={self.x: input_sent, self.encoder_lengths: [len_inputs]})
                    predict_sent = []
                    for word in onehot_pred[0]:
                        predict_sent.append(self.index2word[word])
                        #predict_sent.append(self.embedding_model.wv.index2word[word])
                    #onehot_pred = self.embedding_model.wv.index2word[onehot_pred[0]]
                    for word in predict_sent:
                        output_sent +=  " %s" % (word)
                    #input_sent = input_sent[0][1:]
                    #input_sent.append(self.embedding_model.vocab[onehot_pred].index)
                    #input_sent = [input_sent]
                    print(output_sent)
                except Exception as e:
                    print(e)

if __name__ == '__main__':
    args = sys.argv[1:]
    learning_rate = 0.001
    training_iters = 10000
    display_step = 1000
    n_input = 4
    batch_size = 1
    n_hidden = 300
    path_to_model = "simpleRNN/models/"
    model_name = "seq2seq_model"
    if len(args) >= 1 and args[0] == "train":
        rnn = SimpleRNN(learning_rate, training_iters, display_step, n_input,
            batch_size, n_hidden, path_to_model, model_name, to_train=True)
        rnn.train()
    else:
        rnn = SimpleRNN(learning_rate, training_iters, display_step, n_input,
            1, n_hidden, path_to_model, model_name, to_train=False)
        rnn.run()
