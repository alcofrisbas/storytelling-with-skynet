import os
import csv
definitions = []
with open("data/vocab.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for word in reader:
        if len(word) > 1:
            definitions.append(word)
with open("data/templates.csv", "w") as templates:
    writer = csv.writer(templates, delimiter= ",")
    with open("train.txt", "r") as train:
        lines = train.readlines()
        for line in lines:
            structure = []
            line = line.strip()
            line = line.split(" ")
            for word in line:
                for definition in definitions:
                    if word == definition[0]:
                        structure.append(definition[1])
                        break
            if len(structure) >= 3:
                writer.writerow(structure)
