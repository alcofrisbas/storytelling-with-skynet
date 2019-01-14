import re
import os

os.chdir("../RNN/")
local = os.getcwd()
os.chdir("../webscrape-gutenberg/")
work = os.getcwd()


with open(local + "/train.txt", "w") as f:
    for file in os.listdir("train"):
        with open(work + '/train/' +  file, "r") as f1:
            text = f1.read()
            text = text.replace("\n", " ")
            text = re.split("(?<!.Mr|.Ms|Mrs)[.]", text)
            for sent in text:
                f.write(sent + ".\n")

with open(local + "/valid.txt", "w") as f:
    for file in os.listdir("valid"):
        with open(work + '/valid/' +  file, "r") as f1:
            text = f1.read()
            text = text.replace("\n", " ")
            text = re.split("(?<!.Mr|.Ms|Mrs)[.]", text)
            for sent in text:
                f.write(sent + ".\n")

print(local)
with open(local + "/test.txt", "w") as f:
    for file in os.listdir("test"):
        with open(work + '/test/' +  file, "r") as f1:
            text = f1.read()
            text = text.replace("\n", " ")
            text = re.split("(?<!.Mr|.Ms|Mrs)[.]", text)
            for sent in text:
                f.write(sent + ".\n")
