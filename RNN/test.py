import tensorflow as tf
import numpy as np
import collections

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

checkpoint = tf.train.latest_checkpoint('.\models')

file = 'belling_the_cat.txt'


tf.reset_default_graph()

#tf.placeholder("float", [None, 3, 1])
with tf.Session() as sess:
    data = read_data(file)
    dictionary, reverse_dictionary = build_dataset(data)
    sentence = input("enter:")
    sentence = sentence.strip()
    words = sentence.split(' ')
    symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]

    # import the saved graph
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    # get the graph for this session
    graph = tf.get_default_graph()
    placeholders = tf.contrib.framework.get_placeholders(graph)
    add = sess.graph.get_tensor_by_name("add:0")
    sess.run(tf.global_variables_initializer())
    pred = sess.graph.get_operation_by_name("add")
    for i in range(10):
        keys = np.reshape(np.array(symbols_in_keys, dtype=float), [-1, 3, 1])
        onehot_pred = sess.run(add, {placeholders[0] : keys})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
        symbols_in_keys = symbols_in_keys[1:]
        symbols_in_keys.append(onehot_pred_index)
    print(sentence)

    #print(sess.run(pred, {placeholders[0]: keys}))
    #print(sess.graph.get_operations())
    #print(sess.run(pred, {placeholders[0] : keys}))
    #sess.run(placeholder)
