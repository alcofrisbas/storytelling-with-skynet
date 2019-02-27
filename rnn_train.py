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
        path_to_model=root_path,model_name=model_name, train=train)

if __name__ == '__main__':
    train(training_file="simpleRNN/data/train.txt",root_path="simpleRNN/models/",
        model_name="ptb_model",n_hidden=300, min_count=3,learning_rate=0.001,
        training_iters=1000, n_input=6, batch_size=10,train=True)
