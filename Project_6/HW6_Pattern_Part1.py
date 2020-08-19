from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


def string_gen(k):
    return [a] * k + [N] + [b] * k


def test_string_gen(k):
    return [a] * k + [N]


a = [1.0, 0.0, 0.0, 0.0]
N = [0.0, 1.0, 0.0, 0.0]
b = [0.0, 0.0, 1.0, 0.0]
e = [0.0, 0.0, 0.0, 1.0]
vocabulary = [a, N, b, e]
dictionary = ['a', 'N', 'b', 'e']

# Training Parameters
learning_rate = 0.001
training_steps = 10
display_step = 100
test = 15
training_iter = 200

vocab = sorted(set('aNbe'))

# Network Parameters
num_input = len(vocab)

neuron_num = 10  # Number of LSTM cell neurons.
# char2idx = {u: i for i, u in enumerate(vocab)}
# idx2char = np.array(vocab)

# tf Graph input
X = tf.placeholder("float", [None, None, len(vocab)])
Y = tf.placeholder("float", [None, len(vocab)])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([neuron_num, len(vocab)]))
}
biases = {
    'out': tf.Variable(tf.random_normal([len(vocab)]))
}


# noinspection PyPep8Naming,PyShadowingNames
def RNN(x, weights, biases):

    lstm_cell = tf.nn.rnn_cell.LSTMCell(neuron_num, use_peepholes=True, forget_bias=1.0)
    cell_states = lstm_cell.variables

    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], states, cell_states


logits, states, cell_state = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

cell_1 = tf.summary.scalar('cell_states_neuron_1', states[0][0][0])
cell_2 = tf.summary.scalar('cell_states_neuron_2', states[0][0][1])
cell_3 = tf.summary.scalar('cell_states_neuron_3', states[0][0][2])
cell_4 = tf.summary.scalar('cell_states_neuron_4', states[0][0][3])
cell_5 = tf.summary.scalar('cell_states_neuron_5', states[0][0][4])
cell_6 = tf.summary.scalar('cell_states_neuron_6', states[0][0][5])
cell_7 = tf.summary.scalar('cell_states_neuron_7', states[0][0][6])
cell_8 = tf.summary.scalar('cell_states_neuron_8', states[0][0][7])
cell_9 = tf.summary.scalar('cell_states_neuron_9', states[0][0][8])
cell_10 = tf.summary.scalar('cell_states_neuron_10', states[0][0][9])


merge = tf.summary.merge_all()

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    file_writer = tf.summary.FileWriter('cell_states_summary')

    # ####################### Training ####################### #
    for iteration in range(training_iter):

        for step in range(1, training_steps + 1):

            # batch_x = np.repeat([string_gen(step)], batch_size, axis=0)
            batch_x = np.repeat([string_gen(step)], 1, axis=0)
            batch_y = string_gen(step)[1:] + [e]
            # print("Batch X is: ", batch_x)
            # print("Batch Y is: ", batch_y)
            # print(type(batch_x))
            # print(type(batch_y))
            # print(len(batch_x))
            # print(len(batch_y))

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if iteration % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Iteration: " + str(iteration) + "\tStep " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(
                    loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))
                outputs, Y_res = sess.run((prediction, Y), feed_dict={X: batch_x,
                                                                      Y: batch_y})

                outputs = np.argmax(outputs, 1)
                # print(outputs)
                result = ''
                Y_result = ''
                for i in range(len(outputs)):
                    result += dictionary[outputs[i]]
                    # Y_result += dictionary[int(Y_res[i])]
                # print(result)
                # print(Y_result)
    print("Optimization Finished!")

    # ####################### Testing ####################### #
    for test_num in range(1, test + 1):

        # states_output = sess.run(states, feed_dict={X: np.repeat([test_string_gen(test_num)], 1, axis=0)})
        # states_output = np.array(states_output)
        # print(states_output.shape)

        counter = 0
        input = ''
        result = ''
        for i in range(len(test_string_gen(test_num))):
            # Gets the four element array and returns the corresponding character for printing input (for debugging)
            char = np.argmax(np.array(test_string_gen(test_num)[i]))
            input += dictionary[char]
        print("INPUT IS: ", input)

        outputs = sess.run(prediction, feed_dict={X: np.repeat([test_string_gen(test_num)], 1, axis=0)})
        outputs = np.argmax(outputs, 1)

        for i in range(len(outputs)):
            result += dictionary[outputs[i]]
        # print(result)
        next_input = test_string_gen(test_num)
        while (result[-1] != 'e' or counter == 0) and len(result) <= 2 * test_num + 6:

            counter += 1
            outputs = sess.run(prediction, feed_dict={X: np.repeat([next_input], 1, axis=0)})
            outputs = np.argmax(outputs, 1)
            # print(outputs)
            result = ''
            for i in range(len(outputs)):
                result += dictionary[outputs[i]]

            next_input = next_input + [vocabulary[outputs[-1]]]
        print("K is: " + str(test_num) + "\tResult is:" + result)

    # ####################### Testing for 15 ####################### #

    counter = 0
    input = ''
    result = ''
    for i in range(len(test_string_gen(test))):
        char = np.argmax(np.array(test_string_gen(test)[i]))
        input += dictionary[char]
    # print(input)
    outputs = sess.run(prediction, feed_dict={X: np.repeat([test_string_gen(test)], 1, axis=0)})
    outputs = np.argmax(outputs, 1)

    for i in range(len(outputs)):
        result += dictionary[outputs[i]]
    # print(result)
    next_input = test_string_gen(test)
    while (result[-1] != 'e' or counter == 0) and len(result) <= 2 * test + 6:

        summary = sess.run(merge, feed_dict={X: np.repeat([next_input], 1, axis=0)})
        file_writer.add_summary(summary, counter)

        counter += 1
        outputs = sess.run(prediction, feed_dict={X: np.repeat([next_input], 1, axis=0)})
        outputs = np.argmax(outputs, 1)
        # print(outputs)
        result = ''
        for i in range(len(outputs)):
            result += dictionary[outputs[i]]

        next_input = next_input + [vocabulary[outputs[-1]]]
    print("K is: " + str(test) + "\tResult is:" + result)

# tensorboard --logdir=1:./css1,2:./css2,3:./css3,4:./css4,5:./css5,6:./css6,7:./css7,8:./css8,9:./css9,10:./css10
