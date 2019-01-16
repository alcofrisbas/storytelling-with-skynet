import tensorflow as tf
import numpy as np
import csv

# Initialize writer and creating tfrecord file
writer = tf.python_io.TFRecordWriter("data/train.tfrecords")
def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value for value in values]))
def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value for value in values]))
templates = []
with open("data/templates.csv", "r") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for csvline in csvreader:
        if csvline is not None:
            templates.append(csvline)
with open("data/train.txt.ids", "r") as file:
        lines = file.readlines()
        for place in range(len(lines)-1):
            input_template = templates[place]
            try:
                output_template = templates[place+1]
            except:
                break
            input, output = lines[place], lines[place+1]
            input, output = input.strip('\n'), output.strip('\n')
            input, output = input.split(" "), output.split(" ")
            input, output = np.array(input, dtype=np.int64), np.array(output, dtype=np.int64)
            input_template, output_template = np.array(input_template, dtype=bytes), np.array(output_template, dtype=bytes)
            example = tf.train.Example(features=tf.train.Features(feature={"intput_template": _bytes_feature(input_template), "input": _int64_feature(input), "output":_int64_feature(output), "output_template":_bytes_feature(output_template)}))
            writer.write(example.SerializeToString())
writer.close()
