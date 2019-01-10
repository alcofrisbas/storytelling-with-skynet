from nltk import word_tokenize, pos_tag

def generate_structure(sentence):
    text = word_tokenize(sentence)
    text = pos_tag(text)
    return [i[1] for i in text]

if __name__ == '__main__':
    s = "This is a delicious sentence"
    print(generate_structure(s))
