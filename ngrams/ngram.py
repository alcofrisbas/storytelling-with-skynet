import numpy as np
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
    """
    Check and return
      1. If the prefix exsists in any of the words we added so far
      2. If yes then how may words actually have the prefix
    """
    node = root
    # If the root node has no children, then return False.
    # Because it means we are trying to search in an empty trie
    if not root.children:
        return False, 0
    for word in prefix:
        not_found = True
        # Search through all the children of the present `node`
        for child in node.children:
            if child.key == word:
                # We found the char existing in the child.
                not_found = False
                # Assign node as the child containing the char and break
                node = child
                break
        # Return False anyway when we did not find a char.
        if not_found:
            return False, 0
    # Well, we are here means we have found the prefix. Return true to indicate that
    # And also the counter of the last node. This indicates how many words have this
    # prefix
    return True, node

def predict_next(root, prefix):
    _, parent = find_prefix(root, prefix)
    if _:
        childs = [child.key for child in parent.children]
        total = sum([child.payload for child in parent.children])
        weights = [child.payload/total for child in parent.children]
        print (sum(weights))
        next = np.random.choice(childs, p=weights)
        return next
    return

def corpus_to_dict(fname, n):
    with open(fname, "r") as r:
        text = r.read()
    d = {}
    words = text.split()
    for i in range(len(words)):
        pass

def train(fname, n=4):
    with open(fname) as r:
        s = r.read()
    words = s.lower().split()
    root = Trie("*")
    for i in range(len(words)-n):
        add(root,words[i:i+n])
        if i%1000 == 0:
            print("{} substrings processed".format(str(i)))

        if i > 200000:
            break
    return root

if __name__ == '__main__':
    trie = {}
    root = train("./ngrams/dickens.txt")

    sent = input("enter a sentence: ")
    l = 20
    m = 3
    while sent != "quit":
        sentence = sent.split()
        for i in range(l):
            next = predict_next(root, sentence[-m:])
            sentence.append(next)
            if next[-1] == ".":
                break
        print(" ".join([str(word) for word in sentence]))
        sent = input("enter a sentence: ")
