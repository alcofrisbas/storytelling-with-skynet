import csv
with open("data/vocab.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    with open("dictionary.csv", "r") as dict:
        reader = csv.reader(dict, delimiter=",")
        dictionary = []
        for row in reader:
            dictionary.append(row)
    with open("data/vocab.txt", "r") as vocab:
        text = vocab.readlines()
        for word in text:
            written = False
            word = word.replace("\n", "")
            for row in dictionary:
                if len(row) > 1:
                    if row[0].lower() == word:
                        written = True
                        writer.writerow([word, row[1]])
            if not written:
                #type  = input("What type of word is this? ")
                writer.writerow([word, "(unk.)"])
