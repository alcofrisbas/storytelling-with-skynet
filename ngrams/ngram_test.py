import os, sys
# wrangling modules for the sake of pickle... ugh
# run from root dir
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'./'))
from ngrams import ngram
sys.modules["ngram"] = ngram


def main():
    model = ngram.NGRAM_model("./saves/all_of_lewis.txt", "lewis_model", "./ngrams/models")
    model.m = 2
    if not os.path.isfile("./ngrams/models/lewis_model"):
        #prompt_ngram = ngram.create_model("./saves/all_of_lewis.txt","./ngrams/models/lewis_model", l=1000000)
        model.create_model()

    else:
        #prompt_ngram = ngram.load_model("./ngrams/models/lewis_model")
        model.load_model()
    print("done loading")
    low = 10
    high = 75
    for i in range(10):
        curTopic = ""
        while len(curTopic)< low or len(curTopic) > high:
            curTopic = model.generate_sentence("STOP")
        print("-"*20)
        print(curTopic)

if __name__ == '__main__':
    main()
