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
from tensorflow.contrib import rnn
import random
import collections
import time
from nltk.tokenize import word_tokenize
import gensim
import sys

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/rnn_word'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_file = "RNN/data/train.txt"#'simpleRNN/belling_the_cat.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
#vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 4

# number of units in RNN cell
n_hidden = 300
path_to_model = "RNN/models/"

embedding_model = gensim.models.Word2Vec.load(path_to_model + "my_embedding_model")
vocab_size = len(embedding_model.wv.vocab)
weights = {'out': embedding_model.syn1neg}
# tf Graph input
x = tf.placeholder("float", [None, n_input, n_hidden])
y = tf.placeholder("float", [None, vocab_size])



def RNN(x, weights):
    """
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)
    """
    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    # the shape of outputs is [batch_size, n_input, n_hidden]
    outputs, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs = x, dtype = tf.float32)
    output = states[-1].h
    #outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(output, tf.transpose(weights['out']))

pred = RNN(x, weights)
probas = tf.argmax(pred, 1)
# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
training = False

def generateText(input):
    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint('simpleRNN/models'))
        sentence = input.strip()
        words = sentence.split(' ')
        if len(words) == n_input:
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")


if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        training = True
    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        if training:
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
                symbols = [str(training_data[i]) for i in range(offset, offset+n_input)]
                embedded_symbols = []
                for word in symbols:
                    try:
                        embedding = embedding_model.wv[word]
                    except KeyError:
                        print(word + " not in vocabulary")
                        embedding = np.zeros((300,), dtype=np.float)
                    embedded_symbols.append(embedding)
                # embeded_symbols shape [1, n_input, n_hidden]
                embedded_symbols = [embedded_symbols]

                symbols_out_onehot = np.zeros([1, vocab_size], dtype=float)
                try:
                    symbols_out_onehot[0][embedding_model.wv.vocab[training_data[offset+n_input]].index] = 1.0
                except:
                    symbols_out_onehot[0][0] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])


                _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, probas], \
                                                        feed_dict={x: embedded_symbols, y: symbols_out_onehot})

                onehot_pred = embedding_model.wv.index2word[onehot_pred[0]]

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
                    #symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                    print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,onehot_pred))
                step += 1
                offset += (n_input+1)
            print("Optimization Finished!")
            print("Elapsed time: ", elapsed(time.time() - start_time))
            print("Run on command line.")
            print("\ttensorboard --logdir=%s" % (logs_path))
            print("Point your web browser to: http://localhost:6006/")
            saver = tf.train.Saver()# -*- coding: utf-8 -*-

            saver.save(session, "simpleRNN/models/best_model.ckpt")
        else:
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint('simpleRNN/models'))
        while True:
            prompt = "%s words: " % n_input
            input_sent = input(prompt)
            input_sent = word_tokenize(input_sent)
            embedded_symbols = []
            if len(input_sent) != n_input:
                continue
            try:
                for word in input_sent:
                    try:
                        embedding = embedding_model.wv[word]
                    except KeyError:
                        print(word + " not in vocabulary")
                        embedding = np.zeros((300,), dtype=np.float)
                    embedded_symbols.append(embedding)
                # embeded_symbols shape [1, n_input, n_hidden]
                embedded_symbols = [embedded_symbols]
                output_sent = "%s" % (input_sent)
                for i in range(32):
                    onehot_pred = session.run(probas, feed_dict={x: embedded_symbols})
                    onehot_pred = embedding_model.wv.index2word[onehot_pred[0]]
                    output_sent = output_sent + " %s" % (onehot_pred)
                    embedded_symbols = embedded_symbols[0][1:]
                    embedded_symbols.append(embedding_model.wv[onehot_pred])
                    embedded_symbols = [embedded_symbols]
                print(output_sent)
            except:
                print("Word not in dictionary")
