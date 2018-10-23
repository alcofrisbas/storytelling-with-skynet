import tensorflow as tf
import numpy as np
import collections
import rnn_run
import reader

FLAGS = tf.flags.FLAGS
FLAGS.model = "small"

def sample_from_pmf(probas):
    t = np.cumsum(probas)
    s = np.sum(probas)
    return int(np.searchsorted(t, np.random.rand(1) * s))

def generate_text(sess, model, word_to_index, index_to_word,
    seed='.', n_sentences= 20):
    sentence_cnt = 0
    input_seeds_id = [word_to_index[w] for w in seed.split()]
    state = sess.run(model.initial_state)

    # Initiate network with seeds up to the before last word:
    for x in input_seeds_id[:-1]:
        feed_dict = {model.initial_state: state,
                    model.input.input_data: [[x]]}
        state = sess.run([model.final_state], feed_dict)

    text = seed
    # Generate a new sample from previous, starting at last word seed
    input_id = [[input_seeds_id[-1]]]
    while sentence_cnt < n_sentences:
        feed_dict = {model.input.input_data: input_id,
                    model.initial_state: state}
        probas, state = sess.run([model.probas, model.final_state],
                                feed_dict=feed_dict)
        sampled_word = sample_from_pmf(probas[0])
        if sampled_word == word_to_index['.']:
            text += '.\n'
            sentence_cnt += 1
        else:
            text += ' ' + index_to_word[sampled_word]
            sentence_cnt += 1
        input_wordid = [[sampled_word]]
    print(text)

if __name__ == '__main__':
    word_to_id = reader._build_vocab('ptb.train.txt') # here we load the word -> id dictionnary ()
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys())) # and transform it into id -> word dictionnary
    _, _, test_data, vocab_size = reader.ptb_raw_data()

    eval_config = rnn_run.get_config()
    eval_config.num_steps = 1
    eval_config.batch_size = 1
    model_input = rnn_run.Input(eval_config, test_data)
    sess  = tf.Session()
    initializer = tf.random_uniform_initializer(-eval_config.init_scale,
        eval_config.init_scale)
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        tf.global_variables_initializer()
        mtest = rnn_run.RNNModel(is_training=False, config=eval_config, input_=model_input, vocab_size=vocab_size)
        sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./models'))

    while True:
        sentence = input('Write your sentence: ')
        print(generate_text(sess, mtest, word_to_id, id_to_word, seed=sentence))
        try:
            input('press Enter to continue ... \n')
        except KeyboardInterrupt:
            print('\b\bQuiting now...')
            break
