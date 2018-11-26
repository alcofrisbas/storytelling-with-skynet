import csv
import os

with open('dictionary.csv', "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    dictionary = []
    for file in os.listdir("dictionaries"):
        with open("dictionaries/" + file, "r") as dict:
            reader = csv.reader(dict, delimiter=",")
            for row in reader:
                dictionary.append(row)
    for dict in dictionary:
        for words in dict:
            begin = 0
            word = None
            type = None
            if len(words) > 1:
                for i in range(len(words)):
                    if words[i] == "(":
                        begin = i
                        word = words[:i-2]
                    if words[i] == ")":
                        type = words[begin:i+1]
                        writer.writerow([word, type])
                        break
