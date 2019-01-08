import csv
from nltk.tokenize import word_tokenize
import nltk
wordlist = []
with open("data/vocab.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    with open("ptb.train.txt", "r") as vocab:
        lines = vocab.readlines()
        for line in lines:
            line = line.replace("\n", "")
            text = word_tokenize(line)
            text = nltk.pos_tag(text)
            for word in text:
                if word[0] not in wordlist:
                    wordlist.append(word[0])
                    writer.writerow(word)
