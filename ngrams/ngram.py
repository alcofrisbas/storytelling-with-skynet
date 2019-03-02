import os
import pickle
import time

import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize


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
    sList = [" ".join(word_tokenize(i)) for i in sList]

    with open(fname+".tkn" , 'w') as w:
        w.write(" ".join(sList))

class NGRAM_model:
    def __init__(self, model_path):
        self.models = {}
        self.root = Trie("*")
        self.training_file = ""
        self.model_name = ""
        self.model_path = model_path
        # how many generations on trie
        self.depth = 5
        # how many data-points
        self.l = 100000
        self.display_step = 2000
        self.checkpoints = False
        self.chkpt_step = 100000
        # m-gram on sentence gen
        self.m = 2
        self.sent_length = 200
        # better refine sentences...
        self.low = 10
        self.high = 75

    def get_full_path(self):
        return self.model_path+"/"+self.model_name

    def set_model(self, model_name):
        # fail silently!
        self.model_name = model_name
        if self.models.get(model_name):
            self.root = self.models[model_name]

    def train(self,root):
        """
        trains a ngram trie
        l: max iters
        n: depth
        display_step: change freq of print statements
        """
        with open(self.training_file+".tkn") as r:
            s = r.read(5000000)
        print("done reading files")
        words = s.split()
        words = [w.lower() if w != 'STOP' else w for w in words ]
        print("starting trie construction")
        start = time.time()
        for i in range(len(words)-self.depth):
            add(root,words[i:i+self.depth])
            if i%self.display_step == 0:
                now = time.time()-start
                rate = float(i)/now
                print("{} substrings processed at {}".format(str(i), str(rate)))
            # if self.checkpoints and i % self.chkpt_step == 0:
            #     self.save_model()
            #     self.load_model()
            if i > self.l:
                break
        return root
    def save_model(self,root):
        """
        pickles a model to a file
        """
        print("saving model: {}".format(self.model_path+"/"+self.model_name))
        with open(self.model_path+"/"+self.model_name,'wb') as p:
            pickle.dump(root, p)
        print("done saving")

    def load_model(self, model_name):
        """
        loads a pickled model;
        way faster than retraining
        """
        start = time.time()
        self.model_name = model_name
        with open(self.model_path+"/"+self.model_name, 'rb') as r:
            self.models[model_name] = pickle.load(r, fix_imports=True, encoding='bytes')
        print("done loading model: {}, elapsed: {}s".format(model_name, str(int(time.time()-start))))

    def generate_sentence(self, sent:str):
        """
        Generates a whole sentence
        l: max length of sentence
        m: gram depth
        """
        if sent[-1] == ".":
            sent = sent[:-1]+ " STOP"
        sentence = word_tokenize(sent)
        cut = len(sentence)
        for i in range(self.sent_length):
            next = predict_next(self.root, sentence[-self.m:])
            sentence.append(next)
            if sentence[-1] == "STOP":
                break
        for i in range(len(sentence)):
            if sentence[i] == "i":
                sentence[i] = "I"
        outSent = " ".join([str(word) for word in sentence[cut:-1]])+"."
        outSent = outSent[0].upper()+outSent[1:]
        outSent = outSent.strip().replace('“','').replace('”','').replace('`',"'").replace('"','')
        # ensure there's only one period at the end of the sentence
        if len(outSent) <= 1:
            outSent = self.generate_sentence(sent)
        return self.format_sent(outSent)

    def format_sent(self, outSent):
        while outSent[-1] == ".":
            outSent = outSent[:-1]
        ind = 0
        s = ""
        cap = False
        while ind < len(outSent):
            if outSent[ind] == " ":
                if ind < len(outSent)-1 and outSent[ind+1].isalnum():
                    s += " "
            else:
                if outSent[ind].isalpha() and not cap:
                    s += outSent[ind].upper()
                    cap = True
                else:
                    s += outSent[ind]
            ind += 1
        s = '"'.join(s.split('""'))
        s = "n't".join(s.split(" n't"))
        if s[-1] not in "?!":
            s += "."
        if s[0] == "'":
            s = s[1:]
        return s

    def generate_with_constraints(self, sent:str):
        s = ""
        while len(s)< self.low or len(s) > self.high:
            s = self.generate_sentence(sent)
        return s


    def create_model(self, model_name, data=None):
        """
        slams a whole bunch of methods together so
        you can call one function to create, save, and
        return a model. including data is safe!
        """
        start = time.time()
        print("creating model: {}".format(model_name))
        self.model_name = model_name
        if os.path.exists(self.get_full_path()):
            self.load_model(self.model_name)

            return
        if not data:
            raise Exception("missing arg: data")
        self.training_file = data
        root = Trie("*")
        process_data(self.training_file)
        root = self.train(root)
        self.save_model(root)
        self.models[self.model_name] = root
        print("done creating model: {}, elapsed: {}s".format(model_name, str(int(time.time()-start))))


if __name__ == '__main__':
    pass
