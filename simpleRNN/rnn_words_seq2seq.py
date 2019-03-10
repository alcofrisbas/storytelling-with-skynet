
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


class SimpleRNN:
    # Parameters
    def __init__(self, d, display_step, path_to_model, model_name, to_train):
        self.__dict__.update(d)
        self.display_step = display_step
        self.path_to_model = path_to_model
        self.model_name = model_name
        self.to_train = to_train
        self.start_time = time.time()
        # Target log path
        self.logs_path = '/tmp/tensorflow/rnn_word'
        self.writer = tf.summary.FileWriter(self.logs_path)

        # Text file containing words for training
        self.training_file = d["training_file"]
        # training .txt file
        self.training_data = self.read_data(self.training_file)
        self.output_seq_length = len(self.training_data[1]) - 1
        print("Loaded training data...")
        # pull embedings from models
        self.embedding_model = gensim.models.Word2Vec.load(self.path_to_model + self.model_name)
        self.input_embedding_matrix = np.load(self.path_to_model + self.model_name + "_input_embedding_model.npy")
        self.output_embedding_matrix = tf.cast(np.load(self.path_to_model + self.model_name + "_output_embedding_model.npy"), tf.float32)

        self.vocab_size = self.input_embedding_matrix.shape[0]
        self.weights = {'out': self.output_embedding_matrix}
        # tf Graph input
        self.padded_lengths = 100 # dependent on dataset
        self.x = tf.placeholder(tf.int32, [self.batch_size, self.padded_lengths])
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.padded_lengths+1])
        # encoder and decoder lengths
        self.encoder_lengths = tf.placeholder(tf.int32, shape=(None,), name="encoder_length")
        self.decoder_lengths = tf.placeholder(tf.int32, shape=(None,), name="decoder_length")
        # tokens that begin and end sentences
        self.start_token = len(self.input_embedding_matrix) - 3 # GO
        self.end_token = len(self.input_embedding_matrix) - 1  # PAD

        # model predictions
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
            if len(sent) <= 99:
                sent = word_tokenize(sent)
                sentences.append(sent)
        return sentences

    # core rnn calculations
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
                sequence_length=self.encoder_lengths,time_major= False, dtype=tf.float32)

        with tf.variable_scope("decoding") as decoding_scope:
            self.batch_size = tf.shape(inputs)[0]
            lstm_decoding = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden), rnn.BasicLSTMCell(self.n_hidden)])
            # Attention mechanism and wrapper
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.n_hidden, memory=encoder_outputs,
                memory_sequence_length=self.encoder_lengths)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=lstm_decoding, attention_mechanism=attention_mechanism, attention_layer_size=self.n_hidden)
            initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            output_layer = tf.layers.Dense(len(self.input_embedding_matrix), name="output_projection")
            # different processes for training and inference calculations
            # This was implemented entirely because of attention and how tensorflow handles it
            if self.to_train:
                # get embedding for encoder input
                embedded_output = tf.cast(tf.nn.embedding_lookup(self.input_embedding_matrix, targets), tf.float32)
                max_dec_length = tf.reduce_max(self.decoder_lengths+1, name="max_dec_length")
                # helper for training
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=embedded_output, sequence_length=self.decoder_lengths + 1,
                    time_major= False, name="training_helper")
                # the decoder cell
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state, output_layer=output_layer)

                # get outputs of decoder and calculate validation, land loss
                final_outputs, _final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False, impute_finished=True,
                    maximum_iterations=max_dec_length)
                logits = tf.identity(final_outputs.rnn_output, name="logits")

                # slice targets so that GO is no longer the first word
                # this allows us to preform our loss calculations
                targets = tf.slice(targets, [0,0], [-1, max_dec_length], 'targets')

                masks = tf.sequence_mask(self.decoder_lengths+1, max_dec_length, dtype=tf.float32, name="masks")
                batch_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=masks,
                    name="batch_loss")

                valid_predictions = tf.identity(final_outputs.sample_id, name="valid_preds")



            else:
                # convert start token to tenosr to be passed into helper
                start_tokens = tf.tile(tf.constant([self.start_token], dtype=tf.int32), [self.batch_size],
                    name = "start_tokens")
                # a different helper for inference
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

    def train(self):
        with tf.Session() as session:
            session.run(self.init)
            step = 0
            sent_num = 0
            acc_total = 0
            loss_total = 0

            self.writer.add_graph(session.graph)

            # run seq2seq self.training_iters time
            while step < self.training_iters:
                all_pred = []
                # Generate a minibatch. Add some randomness on selection process.
                if sent_num+1 >= len(self.training_data):
                    sent_num = 0
                symbols = self.training_data[sent_num]
                if len(symbols) == 0:
                    symbols = ["PAD"]
                onehot_batch = []
                for word in symbols:
                    # try to convert the word into a one-hot encoding, if word not in vocab word is UNK
                    try:
                        one_hot = self.embedding_model.wv.vocab[word.lower()].index
                    except KeyError:
                        one_hot = len(self.input_embedding_matrix)-2
                    onehot_batch.append(one_hot)
                len_inputs = len(onehot_batch)
                # padding
                while len(onehot_batch) < self.padded_lengths:
                    onehot_batch.append(len(self.input_embedding_matrix)-1)

                onehot_batch = [onehot_batch]

                # targets always starts with the GO character
                targets = [len(self.input_embedding_matrix)-3]
                for word in self.training_data[sent_num+1]:
                    try:
                        # try to convert the word into a one-hot encoding, if word not in vocab word is UNK
                        targets.append(self.embedding_model.wv.vocab[word.lower()].index)
                    except KeyError:
                        targets.append(len(self.input_embedding_matrix)-2)
                len_targets = len(targets)-1
                # padding
                while (len(targets) < self.padded_lengths+1):
                    targets.append(len(self.input_embedding_matrix)-1)
                targets = [targets]
                # run neural network with onehot_batch, targets and their respective lengths
                _, acc, loss, embedding_pred = session.run([self.optimizer, self.accuracy, self.cost, self.probas], \
                                                        feed_dict={self.x: onehot_batch, self.y: targets,
                                                            self.decoder_lengths: [len_targets], self.encoder_lengths: [len_inputs]})
                predictions = []
                for prediction in embedding_pred[0]:
                    predictions.append(self.index2word[prediction])
                all_pred.append(predictions)
                loss_total += loss
                acc_total += acc
                sent_num += 1
                # let user know training info every self.display_step iters
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
                        i = 0
                        j = 0
                        while i < len(self.training_data) and j < len(all_pred):
                            print("%s - [%s] vs [%s]" % (self.training_data[i],self.training_data[i+1], all_pred[j]))
                            i = i + 2
                            j = j + 1
                # save the model every 5000 iters
                if step % 5000 == 0:
                    print("saving checkpoint!")
                    saver = tf.train.Saver()# -*- coding: utf-8 -*-
                    saver.save(session, self.path_to_model+self.model_name)
                step += 1
            print("Optimization Finished!")
            print("Elapsed time: ", self.elapsed(time.time() - self.start_time))
            print("Run on command line.")
            print("\ttensorboard --logdir=%s" % (self.logs_path))
            print("Point your web browser to: http://localhost:6006/")
    # for webapp inference
    def generate_suggestion(self, session, sentence):
        # get input from user
        sentence = sentence.strip()
        input_sent = word_tokenize(sentence)
        embedded_symbols = []
        try:
            # transform user input into one-hot encoding
            for word in input_sent:
                try:
                    embedding = self.embedding_model.wv.vocab[word.lower()].index
                except KeyError:
                    print(word + " not in vocabulary")
                    embedding = len(self.input_embedding_matrix)-2
                embedded_symbols.append(embedding)
            len_inputs = len(embedded_symbols)
            # padding
            while len(embedded_symbols) < self.padded_lengths:
                embedded_symbols.append(len(self.input_embedding_matrix)-3)
            embedded_symbols = [embedded_symbols]
            onehot_pred = 0
            # for inference only encoder input is necessary
            predictions= session.run(self.probas, feed_dict={self.x: embedded_symbols, self.encoder_lengths: [len_inputs]})
            predict_sent = []
            for word in predictions[0]:
                # transform predicted indices to words
                predict_sent.append(self.index2word[word])
            full_pred = []
            for word in predict_sent:
                if word != "GO":
                    full_pred.append(word)

            full_pred.append(".")
            output_sent = ""
            # capitalize first word
            if full_pred[0] == "'" or full_pred[0] == "\"":
                full_pred[1] = full_pred[1].capitalize()
            full_pred[0] = full_pred[0].capitalize()
            # sanity checking on sentence
            for word in full_pred:
                if word == "\",\"":
                    word = ","
                if word == "." or word == "!" or word == "," or word == "?" or word == ";" or word == ":":
                    output_sent += "%s" % (word)
                else:
                    output_sent +=  " %s" % (word)
            return output_sent
        except Exception as e:
            print(e)
            return e

    # for other inference
    def run(self):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(self.path_to_model))
            while True:
                # take user input
                prompt = "Write a sentence: "
                input_sent = input(prompt)
                input_sent = word_tokenize(input_sent)
                len_inputs = len(input_sent)
                embedded_symbols = []
                try:
                    output_sent = "%s" % (input_sent)
                    input_sent_one_hot = []
                    # transform user input into one-hot encoding
                    for word in input_sent:
                        try:
                            input_sent_one_hot.append(self.embedding_model.wv.vocab[word.lower()].index)
                        except KeyError:
                            input_sent_one_hot.append(len(self.input_embedding_matrix)-2)
                    while len(input_sent_one_hot) < self.padded_lengths:
                        input_sent_one_hot.append(len(self.input_embedding_matrix)-3)
                    input_sent = [input_sent_one_hot]
                    # use inference to get prediction
                    onehot_pred = session.run(self.probas, feed_dict={self.x: input_sent, self.encoder_lengths: [len_inputs]})
                                                                        #self.y: outputs, self.decoder_lengths: [len_outputs]})
                    predict_sent = []
                    for word in onehot_pred[0]:
                        predict_sent.append(self.index2word[word])

                    # remove GO character and stop when period appears
                    full_pred = []
                    for word in predict_sent:
                        if word != "GO":
                            full_pred.append(word)

                    full_pred.append(".")
                    # capitalize first word
                    full_pred[0] = full_pred[0].capitalize()
                    for word in full_pred:
                        if word == "\",\"":
                            word = ","
                        if word == "." or word == "!" or word == "," or word == "?" or word == ";" or word == ":":
                            output_sent += "%s" % (word)
                        else:
                            output_sent +=  " %s" % (word)
                    print(output_sent)
                except Exception as e:
                    print(e)


def run(learning_rate, training_iters, n_input, batch_size, n_hidden, path_to_model, model_name, train, training_file):
    d = {"learning_rate": learning_rate, "training_iters" : training_iters,"n_input" : n_input,"batch_size" : batch_size, "n_hidden" : n_hidden, "training_file": training_file}
    display_step = 1000
    # train if told to
    if train:
        rnn = SimpleRNN(d, display_step, path_to_model, model_name, train)
        rnn.train()
    # otherwise we do inference
    else:
        d["batch_size"] = 1
        rnn = SimpleRNN(d, display_step, path_to_model, model_name, train)
        rnn.run()
