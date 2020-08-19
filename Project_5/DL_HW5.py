# Python 2.7
# By: Amin Mansouri

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

# MNIST data input (img shape: 28*28)
n_input = 28

# MNIST total classes (0-9 digits)
n_classes = 10
n_classes_p4 = 2
std = 0.1

training_iterations = 10
learning_rate = 0.01
batch_size = 128
epsilon = 1e-7

weights = {
    'weight_conv_layer_1': tf.get_variable('W0', shape=(5, 5, 1, 64),
                                           initializer=tf.initializers.random_normal(stddev=std)),
    'weight_conv_layer_2': tf.get_variable('W1', shape=(5, 5, 64, 64),
                                           initializer=tf.initializers.random_normal(stddev=std)),
    # 'weight_conv_layer_3': tf.get_variable('W2', shape=(3, 3, 64, 128),
    # initializer=tf.contrib.layers.xavier_initializer()),
    'w_fc': tf.get_variable('W2', shape=(7 * 7 * 64, 256), initializer=tf.initializers.random_normal(stddev=std)),
    'out': tf.get_variable('W3', shape=(256, n_classes), initializer=tf.initializers.random_normal(stddev=std)),
    'out_p4': tf.get_variable('W4', shape=(256, n_classes_p4), initializer=tf.initializers.random_normal(stddev=std)),
}
biases = {
    'bias_conv_layer_1': tf.get_variable('B0', shape=64, initializer=tf.initializers.random_normal(stddev=std)),
    'bias_conv_layer_2': tf.get_variable('B1', shape=64, initializer=tf.initializers.random_normal(stddev=std)),
    # 'bias_conv_layer_3': tf.get_variable('B2', shape=128, initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc': tf.get_variable('B2', shape=256, initializer=tf.initializers.random_normal(stddev=std)),
    'out': tf.get_variable('B3', shape=n_classes, initializer=tf.initializers.random_normal(stddev=std)),
    'out_p4': tf.get_variable('B4', shape=n_classes_p4, initializer=tf.initializers.random_normal(stddev=std)),
}

bn_gamma_1 = tf.get_variable('gamma_1', shape=64, initializer=tf.initializers.random_normal(stddev=std))
bn_gamma_2 = tf.get_variable('gamma_2', shape=64, initializer=tf.initializers.random_normal(stddev=std))
bn_gamma_3 = tf.get_variable('gamma_3', shape=256, initializer=tf.initializers.random_normal(stddev=std))

bn_beta_1 = tf.get_variable('beta_1', shape=64, initializer=tf.initializers.random_normal(stddev=std))
bn_beta_2 = tf.get_variable('beta_2', shape=64, initializer=tf.initializers.random_normal(stddev=std))
bn_beta_3 = tf.get_variable('beta_3', shape=256, initializer=tf.initializers.random_normal(stddev=std))

# both placeholders are of type float
x = tf.placeholder("float", [None, 28, 28, 1])
# x = tf.placeholder("float", [28, 28])
y = tf.placeholder("float", [None, n_classes])
y_trans_learn = tf.placeholder("float", [None, n_classes_p4])
keep_prob = tf.placeholder("float")
dropout_mode = tf.placeholder("float")


# noinspection PyShadowingNames
def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and ReLU activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def batch_normalization(x, gamma, beta):

    if len(x.shape) >= 3:

        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)

    else:

        batch_mean, batch_var = tf.nn.moments(x, axes=[0], keep_dims=True)

    batch_normal = (x - batch_mean) / tf.sqrt(epsilon + batch_var)

    bn_output = tf.add(tf.multiply(batch_normal, gamma), beta)

    return bn_output


def dropout(x, prob, mode_select):

    prob_tensor_uniform = tf.random.uniform(shape=tf.shape(x), dtype=tf.float32)
    bernoulli_tensor = prob_tensor_uniform > prob
    bernoulli_tensor = tf.to_float(bernoulli_tensor)
    dropout_output = tf.multiply(x, bernoulli_tensor)
    return_1 = dropout_output

    dropout_output = tf.multiply(x, prob)
    return_2 = dropout_output

    dropout_output_o = tf.cond(mode_select > 0, lambda: return_1, lambda: return_2)

    return dropout_output_o

with tf.name_scope(name="Convolution_Layer_1"):
    # tf.nn.dropout(x, keep_prob=keep_prob)
    # here we call the conv2d function we had defined above and pass the input image x, weights weight_conv_layer_1
    # and bias bias_conv_layer_1.
    conv1 = conv2d(x, weights['weight_conv_layer_1'], biases['bias_conv_layer_1'])

    # Batch Normalization

    conv1_output = batch_normalization(conv1, gamma=bn_gamma_1, beta=bn_beta_1)

    # ReLU
    conv1 = tf.nn.relu(conv1_output)

    # Dropout
    # conv1 = dropout(conv1, keep_prob=keep_prob, mode=dropout_mode)
    conv1 = dropout(conv1, keep_prob, dropout_mode)
    # conv1 = dropout(conv1, keep_prob)

with tf.name_scope(name="Max_Pooling_Layer_1"):
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1_maxpool = maxpool2d(conv1, k=2)

# print(x.shape, weights['weight_conv_layer_1'].shape, biases['bias_conv_layer_1'].shape, conv1.shape)


with tf.name_scope(name='Convolution_Layer_2'):
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1_maxpool, weights['weight_conv_layer_2'], biases['bias_conv_layer_2'])

    # Batch Normalization

    conv2_output = batch_normalization(conv2, gamma=bn_gamma_2, beta=bn_beta_2)

    # ReLU
    conv2 = tf.nn.relu(conv2_output)

    # Dropout
    # conv2 = dropout(conv2, keep_prob=keep_prob, mode=dropout_mode)
    conv2 = dropout(conv2, keep_prob, dropout_mode)
    # conv2 = dropout(conv2, keep_prob)

with tf.name_scope(name="Max_Pooling_Layer_2"):
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2_maxpool = maxpool2d(conv2, k=2)

print(weights['weight_conv_layer_2'].shape, biases['bias_conv_layer_2'].shape, conv2_maxpool.shape)

with tf.name_scope(name="Fully_Connected_Layer"):
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2_maxpool, [-1, weights['w_fc'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w_fc']), biases['b_fc'])

    # Batch Normalization

    fc_output = batch_normalization(fc1, gamma=bn_gamma_3, beta=bn_beta_3)

    fc1 = tf.nn.relu(fc1)

    # Dropout
    # fc1 = dropout(fc1, keep_prob=keep_prob, mode=dropout_mode)
    fc1 = dropout(fc1, keep_prob, dropout_mode)
    # fc1 = dropout(fc1, keep_prob)

with tf.name_scope(name="Output_Layer"):
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

data = input_data.read_data_sets('mnist', one_hot=True)

print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1, 28, 28, 1)

print([train_X.shape, test_X.shape])

train_y = data.train.labels
test_y = data.test.labels

print(train_y.shape, test_y.shape)

with tf.name_scope(name="Cost_Calculation"):
    prediction = out

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

with tf.name_scope(name="Optimization"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope(name="Accuracy"):
    # Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled
    # image. and both will be a column vector.
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    # calculate accuracy across all the given images and average them out.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summary settings:

train_loss_summary = tf.summary.scalar('loss_train', cost)
train_accuracy_summary = tf.summary.scalar('train_accuracy', accuracy)
train_summaries = tf.summary.merge([train_loss_summary, train_accuracy_summary])
# file_writer = tf.summary.FileWriter('./Output', sess.graph)

validation_loss_summary = tf.summary.scalar('loss_validation', cost)
validation_accuracy_summary = tf.summary.scalar('validation_accuracy', accuracy)
validation_summaries = tf.summary.merge([validation_loss_summary, validation_accuracy_summary])
# validation_file_writer = tf.summary.FileWriter(output_folder)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Batch Mode Learning

with tf.Session() as sess:
    sess.run(init)

    merge = tf.summary.merge_all()
    summary_writer_train = tf.summary.FileWriter('./Output_train', sess.graph)
    summary_writer_validation = tf.summary.FileWriter('./Output_validation', sess.graph)
    summary_writer_conv_layer_1 = tf.summary.FileWriter('./conv_1_filters')
    summary_writer_conv_layer_2 = tf.summary.FileWriter('./conv_2_filters')

    for i in range(training_iterations):
        for batch in tqdm(range(len(train_X) // batch_size // training_iterations)):
            batch_x = train_X[(batch + i * len(train_X) // batch_size // training_iterations) *
                              batch_size:min((batch + 1 + i * len(train_X) // batch_size // training_iterations) *
                                             batch_size, len(train_X))]
            batch_y = train_y[(batch + i * len(train_X) // batch_size // training_iterations) *
                              batch_size:min((batch + 1 + i * len(train_X) // batch_size // training_iterations) *
                                             batch_size, len(train_y))]

            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                 y: batch_y, keep_prob: 0.5, dropout_mode: 1.0})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y, keep_prob: 0.5, dropout_mode: 1.0})
            #
            # opt = sess.run(optimizer, feed_dict={x: batch_x,
            #                                      y: batch_y, keep_prob: 0.5})
            # loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
            #                                                   y: batch_y, keep_prob: 0.5})
            #
            summary_writer_train.add_summary(
                (sess.run(train_summaries, feed_dict={x: batch_x,
                                                      y: batch_y, keep_prob: 0.5, dropout_mode: 1.0})), batch)

            print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
                  )
            print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y, keep_prob: 0.5,
                                                                     dropout_mode: -1.0})
        print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(valid_loss) + ", Testing Accuracy= " +
              "{:.5f}".format(test_acc)
              )

