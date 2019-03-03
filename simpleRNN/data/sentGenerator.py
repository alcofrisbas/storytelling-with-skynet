
from nltk.tokenize import sent_tokenize, word_tokenize


file_to_read = "simpleRNN/data/charles_dickens_great_expectations.txt"
file_to_write = "simpleRNN/data/charles_dickens_great_expectations1.txt"
#accepts a filename and returns a string of contents
def read(fname):
    with open(fname) as f:
        fileList = f.readlines()
    file_content = " ".join(str(x) for x in fileList)
    return file_content

file_content = read(file_to_read)

#get list of sentences
temp_sentences = [word_tokenize(t) for t in sent_tokenize(file_content)]

with open(file_to_write, "w") as f2:
    sentence = ""
    for sent in temp_sentences:
        line = ""
        for word in sent:
            line = line + word +" "
        line = line + "\n"
        f2.write(line)
