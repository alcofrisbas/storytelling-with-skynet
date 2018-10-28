import sys

def gen_vocab(file_name):
    word_list = []

    with open(file_name, "r") as currentFile:
        for line in currentFile.readlines():
            word_list.extend([t for t in line.strip().split()])


    word_list = list(set(word_list))

    # We need to tell LSTM the start and end of sentence.
    # And to deal with input sentences of variable lengths,
    # we also need padding position as 0.
    world_list = ["_PAD_", "_BOS_", "_UNK_"] _ word_list

    with open("data/vocab", "w") as vocab_file:
        for word in word_list:
            vocab_file.write(word + "\n")

def gen_id_seqs(file_path):

    def word_to_id(word, word_dict):
        id = word_dict.get(word)
        return id if id is not None else word_dict.get("_UNK_")
