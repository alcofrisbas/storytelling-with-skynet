import numpy as np
from nltk.tokenize import sent_tokenize
import pickle
import time

class Trie:
    def __init__(self, key):
        self.key = key
        self.payload = 1.0
        self.children = []
        self.done = False

    def __str__(self):
        return("{}\t{}".format(self.key, self.payload))

def add(root, l):
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
    found = False
    while not found:
        print(prefix)
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
    with open(fname, 'r') as r:
        text = r.read()
    sList = sent_tokenize(text)
    sList = [i[:-1]+" STOP" for i in sList]

    with open(fname+".tkn" , 'w') as w:
        w.write(" ".join(sList))


def train(fname, n=5, l=200000):
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
        if i%2000 == 0:
            now = time.time()-start
            rate = float(i)/now
            print("{} substrings processed at {}".format(str(i), str(rate)))

        if i > l:
            break
    return root

def generate_sentence(root:Trie, sent:str, l=200, m=3):
    sentence = sent.split()
    for i in range(l):
        next = predict_next(root, sentence[-m:])
        sentence.append(next)
        if sentence[-1] == "STOP":
            break
    return " ".join([str(word) for word in sentence[:-1]])

if __name__ == '__main__':
    process_data("./ngrams/dickens.txt")
    root = train("./ngrams/dickens.txt.tkn",l=50000)
    with open("./ngrams/testRoot",'wb') as p:
        pickle.dump(root, p)

    sent = input("enter a sentence: ").lower()
    # cap length of sentence
    l = 200
    # n-gram... after 3, it parrots
    m = 3
    while sent != "quit":
        print (generate_sentence(root, sent))
        sent = input("enter a sentence: ").lower()
