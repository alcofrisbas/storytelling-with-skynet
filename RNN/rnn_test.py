import tensorflow as tf
import numpy as np
import RNN
import re
import os
import csv
import random
import nltk
import grammarUtils


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000



def generate_text(sess, model, word_to_index, index_to_word,
    seed='.', n_sentences= 20):
    # temporarily choosing templates randomly, in future should be smarter
    template = grammarUtils.pick_structure()
    sentence_cnt = 0
    input_seeds_id = []
    seed = seed.lower()
    seed = re.split(r'([;|, |.|,|:|?|!])', seed)
    for w in seed:
        try:
            input_seeds_id.append(word_to_index[w])
        except:   # if word is not in vocabulary, processed as _UNK_
            input_seeds_id.append(word_to_index["_UNK_"])
        print(input_seeds_id)
    state = sess.run(model.initial_state)
    text = ''
    # Generate a new sample from previous, starting at last word seed
    input_id = [[input_seeds_id[-1]]]
    first_word = True
    print(template)
    for POS in template:
        print(POS)
        feed_dict = {model.input_batch: input_id}
        probas= sess.run([model.probas],
                                feed_dict=feed_dict)
        # Want to find the highest probability target with type POS
        print(probas)
        test_probas = probas[0][0]
        test_probas.sort()
        length = len(test_probas)-1
        best_choice = None
        best_choice_idx = None
        for i in range(len(test_probas)):
            word = test_probas[length-i]
            word_idx = np.where(probas==word)[-1][0]
            word = index_to_word[word_idx]
            tag = nltk.pos_tag([word])[0][-1]
            if tag == POS:
                best_choice = word
                best_choice_idx = word_idx
                break
        if first_word:
            text += best_choice.capitalize()
            first_word = False
        else:
            text += ' ' + best_choice
        input_wordid = [[best_choice_idx]]
    print(text)
    return text


def load_model(save=False):
    with open(RNN.FLAGS.vocab_file, "r") as vocab_file:
        reader = csv.reader(vocab_file, delimiter=',')
        for row in reader:
            try:
                lines.append(row[0])
            except:
                pass
        vocab_size = len(lines)
        word_to_id = dict([(b,a) for (a,b) in enumerate(lines)])
        id_to_word = dict([(a,b) for (a,b) in enumerate(lines)])

    eval_config = SmallConfig()
    eval_config.num_steps = 1
    eval_config.batch_size = 1
    sess = tf.Session()
    model = RNN.RNNModel(vocab_size=vocab_size,config=eval_config,
        num_train_samples=1, num_valid_samples=1)
    sess.run(tf.global_variables_initializer())
    if save:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('RNN/models'))
    return sess, model, word_to_id, id_to_word


if __name__ == '__main__':
    sess, model, word_to_id, id_to_word = load_model()

    while True:
        sentence = input('Write your sentence: ')
        #try:
        generate_text(sess, model, word_to_id, id_to_word, seed=sentence)
        #except:
            #print("Word not in dictionary.")
        try:
            input('press Enter to continue ... \n')
        except KeyboardInterrupt:
            print('\b\bQuitting now...')
            break
