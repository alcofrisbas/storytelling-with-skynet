import os, sys
# wrangling modules for the sake of pickle... ugh
# run from root dir
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'./'))
from ngrams import ngram
sys.modules["ngram"] = ngram


def main():
    if not os.path.isfile("./ngrams/models/lewis_model"):
        prompt_ngram = ngram.create_model("./saves/all_of_lewis.txt","./ngrams/models/lewis_model", l=1000000)
    else:
        prompt_ngram = ngram.load_model("./ngrams/models/lewis_model")

    for i in range(10):
        curTopic = ngram.generate_sentence(prompt_ngram, "STOP")
        print("-"*20)
        print(curTopic)

if __name__ == '__main__':
    main()
