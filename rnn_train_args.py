import argparse
import sys

def main():
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
    train=True


    args = parser.parse_args(sys.argv[1:])
    print(args)

if __name__ == '__main__':
    main()
