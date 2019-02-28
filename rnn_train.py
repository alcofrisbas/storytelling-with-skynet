import sys
from pathlib import Path
import argparse


import simpleRNN.rnn_words as rnn_words
import word2vec_gensim as word2vec

def train(training_file, root_path, model_name, n_hidden, min_count, learning_rate, training_iters, n_input, batch_size):
    my_file = Path(root_path + model_name + "_vocab.csv")
    if not my_file.is_file():
        word2vec.create_embedding(training_file=training_file,
            root_path=root_path, model_name=model_name,
            n_hidden=n_hidden, min_count=min_count)

    rnn_words.run(learning_rate=learning_rate, training_iters=training_iters,
        n_input=n_input, batch_size=batch_size,n_hidden=n_hidden,
        path_to_model=root_path,model_name=model_name, train=True, training_file=training_file)

if __name__ == '__main__':
<<<<<<< HEAD
    if len(sys.argv) <= 1:
        training_file="simpleRNN/data/train.txt"
        model_name="basic_model"
    else:
        training_file = sys.argv[1]
        model_name = sys.argv[2]
    train(training_file=training_file,root_path="simpleRNN/models/",
        model_name=model_name,n_hidden=300, min_count=3,learning_rate=0.001,
        training_iters=100000, n_input=6, batch_size=10,train=False)
=======
    parser = argparse.ArgumentParser(description="asdf wtf")
    parser.add_argument("--training-file", "-t", action="store")
    parser.add_argument("--model-name", "-m", action="store")
    parser.add_argument("--n-hidden", "-n", action="store")
    parser.add_argument("--min-count", "-c", action="store")
    parser.add_argument("--learning-rate", "-l", action="store")
    parser.add_argument("--training-iters", "-i", action="store")
    parser.add_argument("--n_input", "-p", action="store")
    parser.add_argument("--batch-size", "-b", action="store")


    training_file="simpleRNN/data/train.txt"
    model_name="basic_model"

    n_hidden=300
    min_count=3
    learning_rate=0.001
    training_iters=1000
    n_input=6
    batch_size=10


    args = parser.parse_args(sys.argv[1:])

    if args.training_file:
        training_file = args.training_file
    if args.model_name:
        model_name = args.model_name
    if args.n_hidden:
        n_hidden = args.n_hidden
    if args.min_count:
        min_count = args.min_count
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.training_iters:
        training_iters = args.training_iters
    if args.n_input:
        n_input = args.n_input
    if args.batch_size:
        batch_size = args.batch_size


    train(training_file,"simpleRNN/models/", model_name, n_hidden, min_count, learning_rate, training_iters, n_input, batch_size)
>>>>>>> d9911da396f18dce8c851f8a3496bb9f15498f82
