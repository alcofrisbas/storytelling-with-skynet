import re
import os

os.chdir("../webscrape-gutenberg/")
work = os.getcwd()


with open("train.txt", "wb") as f:
    for file in os.listdir("train"):
        with open(work + '/train/' +  file, "rb") as f1:
            text = f1.read()
            text = text.strip(b'\r')
            text = text.strip(b'\n')
            for sent in text:
                f.write(sent + "\n")

with open("valid.txt", "wb") as f:
    for file in os.listdir("valid"):
        with open(work + '/valid/' + file, "rb") as f1:
            text = f1.read()
            text = text.strip()
            text = re.split('.', text)
            for sent in text:
                f.write(sent + "\n")

with open(work + "test.txt", "wb") as f:
    for file in os.listdir("test"):
        with open(work + '/test/' + file, "rb") as f1:
            text = f1.read()
            text = text.strip()
            text = re.split('.', text)
            for sent in text:
                f.write(sent + "\n")
