import sys
from pathlib import Path


import simpleRNN.rnn_words as rnn_words
import word2vec_gensim as word2vec

def train(training_file, root_path, model_name, n_hidden, min_count, learning_rate, training_iters, n_input, batch_size,
            train):
    my_file = Path(root_path + model_name + "_vocab.csv")
    if not my_file.is_file():
        word2vec.create_embedding(training_file=training_file,
            root_path=root_path, model_name=model_name,
            n_hidden=n_hidden, min_count=min_count)

    rnn_words.run(learning_rate=learning_rate, training_iters=training_iters,
        n_input=n_input, batch_size=batch_size,n_hidden=n_hidden,
        path_to_model=root_path,model_name=model_name, train=train, training_file=training_file)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        training_file="simpleRNN/data/train.txt"
        model_name="basic_model"
    else:
        training_file = sys.argv[1]
        model_name = sys.argv[2]
    train(training_file=training_file,root_path="simpleRNN/models/",
        model_name=model_name,n_hidden=300, min_count=3,learning_rate=0.001,
        training_iters=100000, n_input=6, batch_size=10,train=False)
