import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import logging

training_file = 'RNN/data/test.txt'
root_path = "RNN/models/"
vector_dim = 300
TENSORBOARD_FILES_PATH = root_path+"/tensorboard"

#accepts a filename and returns a string of contents
def read(fname):
    with open(fname) as f:
        fileList = f.readlines()
    file_content = " ".join(str(x) for x in fileList)
    return file_content

file_content = read(training_file)


#raw sentences is a list of sentences.
raw_sentences = file_content.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# model gets trained in gensim
model = gensim.models.Word2Vec(sentences, iter=100, min_count=1, size=300, workers=10)

#returns the list of indexes of each owrd in the word vector vocab
def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data
index_data = convert_data_to_index(file_content, model.wv)

#convert file to list of words
file_content = file_content.split()

vocab_size = len(model.wv.vocab)
len  = 109
if len < vocab_size:
    word = model.wv.index2word[len]
    vec = model.wv[word]
    #print(vec)
else:
    print("not in vocabulary")
    unk = np.zeros((300,), dtype=np.float)
    print(unk)
    print(unk.shape)

#saves the model to be reused
model.save(root_path + "my_embedding_model")
model = gensim.models.Word2Vec.load(root_path + "my_embedding_model")

# print(model.wv.syn0) #prints input embedding
# print(model.syn1neg) #prints output embedding


# convert the wv word vectors into a numpy matrix that is suitable for insertion
# into our TensorFlow model
embedding_matrix = np.zeros((vocab_size, vector_dim))
for i in range(vocab_size):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# embedding layer weights are frozen to avoid updating embeddings while training
saved_embeddings = tf.constant(embedding_matrix)
embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)
"""
tsv_file_path = root_path +"tensorboard/metadata.tsv"
with open(tsv_file_path,'w+', encoding='utf-8') as file_metadata:
    for i,word in enumerate(model.wv.index2word[:vocab_size]):
        embedding_matrix[i] = model.wv[word]
        file_metadata.write(word+'\n')


#Tensorflow Placeholders
X_init = tf.placeholder(tf.float32, shape=(vocab_size, vector_dim), name="embedding")
X = tf.Variable(X_init)


#Initializer
init = tf.global_variables_initializer()

#Start Tensorflow Session
sess = tf.Session()
sess.run(init, feed_dict={X_init: embedding_matrix})

#Instance of Saver, save the graph.
saver = tf.train.Saver()
writer = tf.summary.FileWriter(TENSORBOARD_FILES_PATH, sess.graph)


#Configure a Tensorflow Projector
config = projector.ProjectorConfig()
embeds = config.embeddings.add()
embeds.metadata_path = tsv_file_path

#Write a projector_config
projector.visualize_embeddings(writer,config)

#save a checkpoint
saver.save(sess, TENSORBOARD_FILES_PATH+'/model.ckpt', global_step = vocab_size)

#close the session
sess.close()
"""
#use command: python -m tensorboard.main --logdir=/Users/tenzindophen/Desktop/StoryBot/tensorboard to get the projector
