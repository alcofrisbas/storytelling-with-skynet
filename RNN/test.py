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

checkpoint = tf.train.latest_checkpoint("models\\")
file = 'belling_the_cat.txt'
#data = read_data(file)
#dictionary, reverse_dictionary = build_dataset(data)
tf.reset_default_graph()
#sentence = input("enter:")
#sentence = sentence.strip()
#sentence = sentence.split(' ')
#symbols_in_keys = [dictionary[str(sentence[i])] for i in range(len(sentence))]
#keys = np.reshape(np.array(symbols_in_keys, dtype=float), [-1, 3, 1])
#keys = tf.convert_to_tensor(keys, dtype=float)
#keys = keys.astype(float)

#keys = [[symbols_in_keys[0]],[symbols_in_keys[1]],[symbols_in_keys[2]]]



#tf.placeholder("float", [None, 3, 1])
with tf.Session() as sess:
    # import the saved graph
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    # get the graph for this session
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    placeholder = sess.graph.get_operation_by_name("Placeholder")
    x = tf.placeholder(float, [None, 3, 1])
    print(sess.graph)
    pred = sess.graph.get_operation_by_name("add")

    #sess.run(placeholder)
