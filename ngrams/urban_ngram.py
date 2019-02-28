from ngram import *


def scrape_urban(fname):
    l = []
    with open(fname, 'r') as r:
        with open(fname+".tkn", 'w') as w:
            for line in r:
                if('"') in line:
                    w.write(line.split('"')[1]+" STOP\n")
                    #l.append(line.split('"')[1])

def main():
    trie = {}
    process_data("./ngrams/dickens.txt")
    root = train("./ngrams/dickens.txt.tkn")

    sent = input("enter a sentence: ").lower()
    # cap length of sentence
    l = 200
    # n-gram... after 3, it parrots
    m = 3
    while sent != "quit":
        print (generate_sentence(root, sent))
        sent = input("enter a sentence: ").lower()


if __name__ == '__main__':
    #scrape_urban("./ngrams/urbandict-word-def.csv")
    root = train("./ngrams/urbandict-word-def.csv.tkn",l=50000)

    sent = input("enter a sentence: ").lower()
    # cap length of sentence
    l = 200
    # n-gram... after 3, it parrots
    m = 3
    while sent != "quit":
        print (generate_sentence(root, sent, m=2))
        sent = input("enter a sentence: ").lower()
