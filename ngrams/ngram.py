import numpy as np
from nltk.tokenize import sent_tokenize
import pickle
import time

class Trie:
    """
    A data structure for storing n-grams
    """
    def __init__(self, key):
        self.key = key
        self.payload = 1.0
        self.children = []
        self.done = False

    def __str__(self):
        return("{}\t{}".format(self.key, self.payload))

def add(root, l):
    """
    adds a word to the ngram trie
    """
    node = root
    for word in l:
        in_child = False
        for child in node.children:
            if child.key == word:
                child.payload += 1
                node = child
                in_child = True
                break
        if not in_child:
            newNode = Trie(word)
            node.children.append(newNode)
            node = newNode
    node.done = True

def find_prefix(root, prefix):
    """
    gets info from trie based on an existing sentence
    """
    node = root
    if not root.children:
        return False, 0
    for word in prefix:
        not_found = True
        for child in node.children:
            if child.key == word:
                not_found = False
                node = child
                break
        if not_found:
            return False, 0

    return True, node

def predict_next(root, prefix):
    """
    given a trie and a sentence, predicts the next word
    """
    found = False
    while not found:
        #print(prefix)
        found, parent = find_prefix(root, prefix)
        if found:
            childs = [child.key for child in parent.children]
            total = sum([child.payload for child in parent.children])
            weights = [child.payload/total for child in parent.children]
            #print (sum(weights))
            next = np.random.choice(childs, p=weights)
            return next
        prefix.pop(0)
    return

def process_data(fname):
    """
    quick and dirty data preprocessor.
    This is bad. But for now, it'll work
    """
    with open(fname, 'r') as r:
        text = r.read()
    sList = sent_tokenize(text)
    sList = [i[:-1]+" STOP" for i in sList]

    with open(fname+".tkn" , 'w') as w:
        w.write(" ".join(sList))


def train(fname, n=5, l=200000,display_step=2000):
    """
    trains a ngram trie
    l: max iters
    n: depth
    display_step: change freq of print statements
    """
    with open(fname) as r:
        s = r.read(1000000)
    print("done reading files")
    words = s.split()
    words = [w.lower() if w != 'STOP' else w for w in words ]
    root = Trie("*")
    print("starting trie construction")
    start = time.time()
    for i in range(len(words)-n):
        add(root,words[i:i+n])
        if i%display_step == 0:
            now = time.time()-start
            rate = float(i)/now
            print("{} substrings processed at {}".format(str(i), str(rate)))

        if i > l:
            break
    return root

def save_model(fname, model):
    """
    pickles a model to a file
    """
    with open(fname,'wb') as p:
        pickle.dump(model, p)

def load_model(fname):
    """
    loads a pickled model;
    way faster than retraining
    """
    with open(fname, 'rb') as r:
        root = pickle.load(r, fix_imports=True, encoding='bytes')
    return root

def generate_sentence(root:Trie, sent:str, l=200, m=3):
    """
    Generates a whole sentence
    l: max length of sentence
    m: gram depth
    """
    if sent[-1] == ".":
        sent = sent[:-1]+ " STOP"
    sentence = sent.split()
    cut = len(sentence)
    for i in range(l):
        next = predict_next(root, sentence[-m:])
        sentence.append(next)
        if sentence[-1] == "STOP":
            break

    outSent = " ".join([str(word) for word in sentence[cut:-1]])+"."
    outSent = outSent[0].upper()+outSent[1:]
    # ensure there's only one period at the end of the sentence
    while outSent[-1] == ".":
        outSent = outSent[:-1]
    return outSent.strip()+"."


def create_model(fname, model_name, depth=5, l=100000):
    """
    slams a whole bunch of methods together so
    you can call one function to create, save, and
    return a model.
    """
    process_data(fname)
    root = train(fname+".tkn",n=depth,l=l)
    save_model(model_name, root)
    return root


if __name__ == '__main__':
    process_data("./ngrams/dickens.txt")
    root = load_model("./ngrams/testRoot")
    #root = train("./ngrams/dickens.txt.tkn",l=50000)
    save_model("./ngrams/testRoot",root)

    sent = input("enter a sentence: ").lower()
    # cap length of sentence
    l = 200
    # n-gram... after 3, it parrots
    m = 3
    while sent != "quit":
        print (generate_sentence(root, sent,m=3))
        sent = input("enter a sentence: ").lower()
