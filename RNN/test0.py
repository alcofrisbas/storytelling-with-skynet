import tensorflow as tf

def parse(line):
    line_split = tf.string_split([line])
    input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
    output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
    return input_seq, output_seq

training_dataset = tf.data.TextLineDataset('data/ptb.test.txt.ids').map(parse).padded_batch(1, padded_shapes=([None],[None]))
dataset0 = training_dataset.skip(1)
print(training_dataset)
print(dataset0)
dataset = tf.data.Dataset.zip((training_dataset, dataset0))
print(dataset)
iterator = tf.data.Iterator.from_structure(dataset.output_types,
    dataset.output_shapes)
#training_init_op = iterator.make_initializer(training_dataset)
#test_iterator = iterator.make_initializer(dataset0)
t_iterator = iterator.make_initializer(dataset)
input, output = iterator.get_next()
i, j = input
print(i)
print(j)
#print(input[0])
with tf.Session() as sess:
    #sess.run(training_init_op)
    #print(sess.run(input))
    #sess.run(test_iterator)
    #print(sess.run(input))
    sess.run(t_iterator)
    print(sess.run(input))
