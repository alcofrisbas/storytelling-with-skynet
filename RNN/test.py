import tensorflow as tf
import numpy as np
import RNN
import re
import os



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


def sample_from_pmf(probas):
    t = np.cumsum(probas)
    s = np.sum(probas)
    return int(np.searchsorted(t, np.random.rand(1) * s))

def generate_text(sess, model, word_to_index, index_to_word,
    seed='.', n_sentences= 20):
    sentence_cnt = 0
    input_seeds_id = []
    seed = seed.lower()
    seed = re.split(r'([;|, |.|,|:|?|!])', seed)
    new_seed = []
    for word in seed:
        if word not in ['', ' ']:
            new_seed.append(word)

    for w in new_seed:
        try:
            input_seeds_id.append(word_to_index[w])
        except:   # if word is not in vocabulary, processed as _UNK_
            input_seeds_id.append(word_to_index["_UNK_"])
    state = sess.run(model.initial_state)

    text = ''
    # Generate a new sample from previous, starting at last word seed
    input_id = [[input_seeds_id[-1]]]
    while sentence_cnt < n_sentences:
        feed_dict = {model.input_batch0: [input_seeds_id[:-1]], model.input_batch1: input_id}
        probas= sess.run([model.probas],
                                feed_dict=feed_dict)

        sampled_word = sample_from_pmf(probas[0])
        punctuation = [word_to_index['.'], word_to_index['?'], word_to_index['!']]
        input_id = [[sampled_word]]
        if sampled_word in punctuation:
            text += '.\n'
            sentence_cnt += 20
        else:
            if index_to_word[sampled_word] in ["_UNK_","_PAD_", "_BOS_", "_EOS_"]:
                text += ''
            else:
                # case if processing the first word
                if text == '':
                    text += index_to_word[sampled_word].capitalize()
                else:
                    text += ' ' + index_to_word[sampled_word]
            sentence_cnt += 1
        input_wordid = [[sampled_word]]
    print(text)
    return text


def load_model():
    with open(RNN.FLAGS.vocab_file, "r") as vocab_file:
        lines = [line.strip() for line in vocab_file.readlines()]
        vocab_size = len(lines)
        word_to_id = dict([(b,a) for (a,b) in enumerate(lines)])
        id_to_word = dict([(a,b) for (a,b) in enumerate(lines)])

    eval_config = SmallConfig()
    eval_config.num_steps = 1
    eval_config.batch_size = 1
    sess = tf.Session()
    model = RNN.RNN.RNNModel(vocab_size=vocab_size,config=eval_config,
        num_train_samples=1, num_valid_samples=1)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('../RNN/model'))
    return sess, model, word_to_id, id_to_word


if __name__ == '__main__':
    with open(RNN.FLAGS.vocab_file, "r") as vocab_file:
        lines = [line.strip() for line in vocab_file.readlines()]
        vocab_size = len(lines)
        word_to_id = dict([(b,a) for (a,b) in enumerate(lines)])
        id_to_word = dict([(a,b) for (a,b) in enumerate(lines)])

    eval_config = SmallConfig()
    eval_config.num_steps = 1
    eval_config.batch_size = 1
    with tf.Session() as sess:
        model = RNN.RNNModel(vocab_size=vocab_size,config=eval_config,
            num_train_samples=1, num_valid_samples=1)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./model'))

        while True:
            sentence = input('Write your sentence: ')
            #try:
            generate_text(sess, model, word_to_id, id_to_word, seed=sentence)
            #except:
                #print("Word not in dictionary.")
            try:
                input('press Enter to continue ... \n')
            except KeyboardInterrupt:
                print('\b\bQuiting now...')
                break
