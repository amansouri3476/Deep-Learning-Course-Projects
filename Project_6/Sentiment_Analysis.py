import gensim
from sklearn.model_selection import train_test_split
from numpy import array, append, zeros, reshape
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# ############################################## Word Embedding Process ##############################################

# Creating a dictionary of phrases and their index

phrase_index_dictionary = {}

with open('dictionary.txt', 'r') as dictionary:
    for line in dictionary:
        phrase, index = line.split('|')
        phrase_index_dictionary[phrase] = int(index)


# Creating a dictionary of phrase indexes and their corresponding score

phraseindex_score_dictionary = {}

with open('sentiment_labels.txt', 'r') as sentiment_labels:
    sentiment_labels.readline()  # skip the first line

    for line in sentiment_labels:
        phraseindex, score = line.split('|')
        phraseindex_score_dictionary[int(phraseindex)] = float(score)


sentences = []
vectorized_sentences = []
sentiment_scores = []

with open('datasetSentences.txt', 'r') as sentences_file:
    sentences_file.readline()  # skip the first line

    for line in sentences_file:

        sentence = line.split('\t')[1]
        sentence = sentence[0:len(sentence) - 1]  # removing \n
        try:
            # phrase_index_dictionary[sentence]  # if the sentence is not available in the dic -> except
            # (this is faster than using if)

            # Extracting words
            sentences.append(sentence.split(' '))
            sentiment_scores.append(phraseindex_score_dictionary[phrase_index_dictionary[sentence]])
            embedded_sentence = []
            for word in sentence:
                try:
                    embedded_sentence.append(model[word])
                    # print('embedded sentence is: ', embedded_sentence)
                except KeyError:
                    pass

            vectorized_sentences.append(np.array(embedded_sentence))

        except KeyError:
            pass

x_train, x_test, y_train, y_test = train_test_split(np.array(vectorized_sentences), np.array(sentiment_scores),
                                                    test_size=0.4)

max_sentence_len = 0
x_train_lengths = []
x_test_lengths = []

for x in [x_train, x_test]:
    for sentence in x:
        max_sentence_len = sentence.shape[0] if sentence.shape[0] > max_sentence_len else max_sentence_len

for i in range(len(x_train)):
    x_train_lengths.append(len(x_train[i]))
    while x_train[i].shape[0] < max_sentence_len:
        x_train[i] = append(x_train[i], zeros([1, len(x_train[i][0])]), axis=0)

for i in range(len(x_test)):
    x_test_lengths.append(len(x_test[i]))
    while len(x_test[i]) < max_sentence_len:
        x_test[i] = append(x_test[i], zeros([1, len(x_test[i][0])]), axis=0)

# #################################################### NETWORK ####################################################


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length_ = tf.reduce_sum(used, 1)
    length_ = tf.cast(length_, tf.int32)
    return length_


def last_relevant(output, length_):
    batch_size_ = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index_f = tf.range(0, batch_size_) * max_length + (length_ - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index_f)
    return relevant


learning_rate = 0.01
hidden_neuron_num = 256
display_step = 100
vocab_size = len(x_train[0][0])
train_epochs = 5
batch_size = 64

# tf Graph input
x = tf.placeholder("float", [None, None, vocab_size])
y = tf.placeholder("float", [None, 1])

# Define weights
weights = tf.Variable(tf.random_normal([hidden_neuron_num, 1]))
biases = tf.Variable(tf.random_normal([1]))


lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_neuron_num)

# Get lstm cell output
outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=length(x))

# prediction = tf.matmul(outputs[:, -1, :], weights) + biases
rnn_prediction = tf.matmul(last_relevant(outputs, length(x)), weights) + biases

cost = tf.losses.mean_squared_error(labels=y, predictions=rnn_prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # ############################################# Training ############################################# #

    for epoch in range(train_epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = array(x_train[i: i + batch_size])
            y_batch = array(y_train[i: i + batch_size])

            reshaped_x_batch = array([x_train[i]])
            for j in range(i + 1, min(i + batch_size, len(x_train))):
                reshaped_x_batch = append(reshaped_x_batch, array([x_train[j]]), axis=0)

            reshaped_y_batch = reshape(y_batch, newshape=[
                len(y_batch), 1
            ])

            _, loss, prediction = sess.run([optimizer, cost, rnn_prediction],
                                           feed_dict={x: reshaped_x_batch,
                                                      y: reshaped_y_batch})
            if i % 20 == 0:
                print('Epoch Number:', epoch, 'Batch:', i)
                print('loss:', loss)

        # ############################################# Testing ############################################# #

    counter = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = array(x_test[i: i + batch_size])
        y_batch = array(y_test[i: i + batch_size])

        reshaped_x_batch = array([x_test[i]])
        for j in range(i + 1, min(i + batch_size, len(x_test))):
            reshaped_x_batch = append(reshaped_x_batch, array([x_test[j]]), axis=0)

        reshaped_y_batch = reshape(y_batch, newshape=[
            len(y_batch), 1
        ])

        loss, prediction = sess.run([cost, rnn_prediction],
                                    feed_dict={x: reshaped_x_batch,
                                               y: reshaped_y_batch})

        if loss < 0.1:
            counter += 1

    print('Test Accuracy: ', counter / len(range(0, len(x_test), batch_size)))
