import os, sys
# wrangling modules for the sake of pickle... ugh
# run from root dir
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'./'))
from ngrams import ngram
sys.modules["ngram"] = ngram


def main():
    model = ngram.NGRAM_model("./ngrams/models")

    model.create_model("lewis_model2")
    model.create_model("5max200000.model")
    model.create_model("dickens_model", data="./simpleRNN/data/all_of_dickens.txt")

    model.set_model("5max200000.model")
    model.m = 2

    for i in range(10):
        curTopic = ""
        curTopic = model.generate_with_constraints("STOP")
        print("-"*20)
        print(curTopic)

if __name__ == '__main__':
    main()
