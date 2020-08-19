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

weights = {
    'weight_conv_layer_1': tf.get_variable('W0', shape=(5, 5, 1, 64),
                                           initializer=tf.initializers.random_normal(stddev=std)
                                           ),
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


# noinspection PyShadowingNames
def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and ReLU activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# both placeholders are of type float
x = tf.placeholder("float", [None, 28, 28, 1])
# x = tf.placeholder("float", [28, 28])
y = tf.placeholder("float", [None, n_classes])
y_trans_learn = tf.placeholder("float", [None, n_classes_p4])

# here we call the conv2d function we had defined above and pass the input image x, weights weight_conv_layer_1
# and bias bias_conv_layer_1.
conv1 = conv2d(x, weights['weight_conv_layer_1'], biases['bias_conv_layer_1'])
# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
conv1_maxpool = maxpool2d(conv1, k=2)

# print(x.shape, weights['weight_conv_layer_1'].shape, biases['bias_conv_layer_1'].shape, conv1.shape)

# Convolution Layer
# here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
conv2 = conv2d(conv1_maxpool, weights['weight_conv_layer_2'], biases['bias_conv_layer_2'])
# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
conv2_maxpool = maxpool2d(conv2, k=2)

print(weights['weight_conv_layer_2'].shape, biases['bias_conv_layer_2'].shape, conv2_maxpool.shape)

# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
fc1 = tf.reshape(conv2_maxpool, [-1, weights['w_fc'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['w_fc']), biases['b_fc'])
fc1 = tf.nn.relu(fc1)
# Output, class prediction
# finally we multiply the fully connected layer with the weights and add a bias term.
out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])


training_iterations = 10
learning_rate = 0.01
batch_size = 128
keep_prob = tf.placeholder("float")

data = input_data.read_data_sets('mnist', one_hot=True)

print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))

curr_img = np.reshape(data.train.images[1], (28, 28))
curr_lbl = np.argmax(data.train.labels[1, :])
plt.imshow(curr_img)
plt.show()

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1, 28, 28, 1)

print([train_X.shape, test_X.shape])

train_y = data.train.labels
test_y = data.test.labels

print(train_y.shape, test_y.shape)

prediction = out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
#  and both will be a column vector.
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
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                 y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})

            summary_writer_train.add_summary(
                (sess.run(train_summaries, feed_dict={x: batch_x,
                                                      y: batch_y})), batch)

            print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
                  )
            print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})

        summary_writer_validation.add_summary(
            (sess.run(validation_summaries, feed_dict={x: test_X,
                                                       y: test_y})), i)

        print("Testing Accuracy:", "{:.5f}".format(test_acc))

        # summary_writer_conv_layer_1.add_summary(sess.run(filter_summary_conv_1, feed_dict={x: test_X,
        #                                                                                    y: test_y}), i)
        # summary_writer_conv_layer_2.add_summary(sess.run(filter_summary_conv_2, feed_dict={x: test_X,
        #                                                                                    y: test_y}), i)

    save_path = saver.save(sess, './tmp/model.ckpt')

    summary_writer_train.close()
    summary_writer_validation.close()

    # Plotting first layer's weights

    weights_image_conv_1 = np.reshape(sess.run(weights['weight_conv_layer_1']).T, newshape=(64, 5, 5))

    fig_1 = plt.figure()
    plt.figure(num=None, figsize=(30, 30), dpi=100)
    plt.title('First Convolution Layer Weights')

    for j in np.arange(0, 64):
        plt.subplot(8, 8, j + 1)
        plt.imshow((weights_image_conv_1[j][:][:]).T)

    plt.show()

    # Plotting first cnn layer's output for class '5'

    feed_x = test_X[15]
    feed_x = feed_x.reshape(-1, 28, 28, 1)
    first_layer_output_class_5 = np.reshape(sess.run(conv1, feed_dict={x: feed_x}).T, newshape=(64, 28, 28))

    fig_2 = plt.figure()
    plt.figure(num=None, figsize=(30, 30), dpi=100)
    plt.title('First Convolution Layer Outputs')

    for j in np.arange(0, 64):
        plt.subplot(8, 8, j + 1)
        plt.imshow((first_layer_output_class_5[j][:][:]).T)

    plt.show()

    # Plotting second cnn layer's output for class '5'

    second_layer_output_class_5 = np.reshape(sess.run(conv2, feed_dict={x: feed_x}).T, newshape=(64, 14, 14))

    fig_3 = plt.figure()
    plt.figure(num=None, figsize=(30, 30), dpi=100)
    plt.title('Second Convolution Layer Outputs')

    for j in np.arange(0, 64):
        plt.subplot(8, 8, j + 1)
        plt.imshow((second_layer_output_class_5[j][:][:]).T)

    plt.show()

# For future use: tensorboard --logdir=train:./Output_train,validation:./Output_validation,conv_1:./conv_1_filters,
# conv2:./conv_2_filters
