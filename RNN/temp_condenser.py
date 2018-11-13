import re
import os

os.chdir("../webscrape-gutenberg/")
work = os.getcwd()

with open("train.txt", "w") as f:
    for file in os.listdir("train"):
        with open(work + '/train/' +  file, "r") as f1:
            sents = []
            lines = f1.readlines()
            for line in lines:
                sents += re.split(r'([.])', line)
            for sent in sents:
                f.write(sent + "\n")

with open("valid.txt", "w") as f:
    for file in os.listdir("valid"):
        with open(work + '/valid/' + file, "r") as f1:
            sents = []
            lines = f1.readlines()
            for line in lines:
                sents += re.split(r'([.])', line)
            for sent in sents:
                f.write(sent + "\n")
with open(word + "test.txt", "w") as f:
    for file in os.listdir("test"):
        with open(workd + '/test/' + file, "r") as f1:
            sents = []
            lines = f1.readlines()
            for line in lines:
                sents += re.split(r'([.])', line)
            for sent in sents:
                f.write(sent + "\n")
