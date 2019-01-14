import re
import os

os.chdir("../webscrape-gutenberg/")
work = os.getcwd()


with open("train.txt", "w") as f:
    for file in os.listdir("train"):
        with open(work + '/train/' +  file, "r") as f1:
            text = f1.read()
            text = text.replace("\n", " ")
            text = re.split("(?<!.Mr|.Ms|Mrs)[.]", text)
            for sent in text:
                f.write(sent + "\n")

with open("valid.txt", "w") as f:
    for file in os.listdir("valid"):
        with open(work + '/valid/' +  file, "r") as f1:
            text = f1.read()
            text = text.replace("\n", " ")
            text = re.split("(?<!.Mr|.Ms|Mrs)[.]", text)
            for sent in text:
                f.write(sent + "\n")


with open("test.txt", "w") as f:
    for file in os.listdir("test"):
        with open(work + '/test/' +  file, "r") as f1:
            text = f1.read()
            text = text.replace("\n", " ")
            text = re.split("(?<!.Mr|.Ms|Mrs)[.]", text)
            for sent in text:
                f.write(sent + "\n")
