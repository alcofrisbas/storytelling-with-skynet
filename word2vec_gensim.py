import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
import csv

def create_embedding(training_file, root_path, model_name,n_hidden, min_count):
    vector_dim = n_hidden
    TENSORBOARD_FILES_PATH = root_path+"/tensorboard"

    #accepts a filename and returns a string of contents
    def read(fname):
        with open(fname) as f:
            fileList = f.readlines()
        file_content = " ".join(str(x) for x in fileList)
        return file_content

    file_content = read(training_file)

    #get list of sentences
    temp_sentences = [word_tokenize(t) for t in sent_tokenize(file_content)]
    sentences = []
    # set all words to lowercase
    max = 0
    for sent in temp_sentences:
        if max < len(sent):
            max = len(sent)
        sent_to_append = []
        for word in sent:
            sent_to_append.append(word.lower())
        sentences.append(sent_to_append)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # model gets trained in gensim
    model = gensim.models.Word2Vec(sentences, iter=100, min_count=min_count, size=n_hidden,  workers=10)
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

    #saves the model to be reused
    model.save(root_path + model_name)
    model = gensim.models.Word2Vec.load(root_path + model_name)

    # print(model.wv.syn0) #prints input embedding
    # print(model.syn1neg) #prints output embedding

    #create a csv file to store vocab
    file_csv = open(root_path + model_name + "_vocab.csv", "w")
    writer = csv.writer(file_csv)
    file_csv.close

    #add special tokens to the vocab
    for word in model.wv.index2word:
        writer.writerow([word])
    writer.writerow(["GO"])
    writer.writerow(["UNK"])
    writer.writerow(["PAD"])


    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # into our TensorFlow model
    embedding_matrix = np.zeros((vocab_size + 3, vector_dim))
    for i in range(vocab_size):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    #add vector of zeros for special tokens in matrix
    embedding_matrix[vocab_size] = np.zeros((vector_dim))
    embedding_matrix[vocab_size + 1] = np.zeros((vector_dim))
    # embedding layer weights are frozen to avoid updating embeddings while training
    saved_embeddings = tf.constant(embedding_matrix)
    embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

    output_embedding_matrix = np.zeros((vocab_size +3, vector_dim))
    for i in range(vocab_size):
        output_embedding_vector = model.syn1neg[i]
        output_embedding_matrix[i] = output_embedding_vector

    output_embedding_matrix[vocab_size] = np.zeros((vector_dim))
    output_embedding_matrix[vocab_size + 1] = np.zeros((vector_dim))

    saved_output_embeddings = tf.constant(output_embedding_matrix)
    output_embedding = tf.Variable(initial_value=saved_output_embeddings, trainable=False)

    #creata a file and store the index to vector matrix
    file_path = root_path + model_name + "_input_embedding_model"
    np.save(root_path+ model_name + "_output_embedding_model", output_embedding_matrix)
    np.save(file_path, embedding_matrix)
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
