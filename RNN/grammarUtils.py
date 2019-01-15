from nltk import word_tokenize, pos_tag
from random import randrange as rrange
import time

"""
given a sentence, returns
a list containing the nltk
"""
def structure_from_sent(sentence):
    text = word_tokenize(sentence)
    text = pos_tag(text)
    return [i[1] for i in text]

"""
given our list of structures, returns
a random sentence structure 
"""
def pick_structure(max=42068):
    start = time.time()
    with open("data/templates.csv", "r") as r:
        for i in range(rrange(0,max)):
            next(r)
        #print(i)
        x = ""
        while x == "":
            x = next(r).strip()
    end = time.time() - start
    print("total time: {}".format(str(end)))
    return x



if __name__ == '__main__':
    s = "This is a delicious sentence"
    print(structure_from_sent(s))
    print(pick_structure())
