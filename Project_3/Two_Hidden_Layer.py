# Python 2.7
# By: Hatef Otroshi
import tensorflow as tf
# import numpy as np
# from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

number_of_neurons_hidden_layer = 50
learning_rate = 0.5
mini_batch_size = 64

mnist = input_data.read_data_sets("mnist", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="input")  # Input array (28*28 = 784)
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="label")  # Output: 10 different classes.

# Adding a name scope ensures logical grouping of the layers in the graph.
with tf.name_scope(name="Hidden_layer"):
    w1 = tf.Variable(tf.random_normal(shape=(784, number_of_neurons_hidden_layer), mean=0, stddev=2, seed=1), name="W")
    b1 = tf.Variable(tf.zeros([number_of_neurons_hidden_layer]), name="B")

    hidden_layer1_input = tf.matmul(x, w1) + b1
    sigmoid1 = tf.tanh(hidden_layer1_input)

    w2 = tf.Variable(tf.random_normal(shape=(number_of_neurons_hidden_layer, number_of_neurons_hidden_layer), mean=0, stddev=2, seed=1), name="W")
    b2 = tf.Variable(tf.zeros([number_of_neurons_hidden_layer]),
                     name="B")

    hidden_layer2_input = tf.matmul(sigmoid1, w2) + b2
    sigmoid2 = tf.tanh(hidden_layer2_input)

with tf.name_scope("output_layer"):
    w3 = tf.Variable(tf.random_normal(shape=(number_of_neurons_hidden_layer, 10), mean=0, stddev=2, seed=1000), name="W")
    b3 = tf.Variable(tf.zeros([10]), name="B")  # 10, is the number of classes since we are classifying the data into 10
    #  different classes as is mentioned in the pdf file.

    outputlayer_input = tf.matmul(sigmoid2, w3) + b3
    output = tf.nn.softmax(outputlayer_input)

with tf.name_scope("xent"):
    # tf.reduce_mean calculates the mean and reduces tha tensor to a single value
    # it is mentioned in the document that the logits argument should not be the output of a softmax. (as opposed here)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    cross_entropy_validation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))


with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    # _y is one-hot coded and tf.argmax(a, 1) returns the index of maximum value in a. The following code indicates
    # where predicted class is the same as label
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))

    # The output of correct_prediction is of type boolean and to be casted to some other type.
    accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy_train")
    accuracy_validation = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy_train")


# Creating nodes to monitor how these variables change over time
# tf.summary.scalar("Cross_Entropy_train", cross_entropy)
# tf.summary.scalar("accuracy_train", accuracy_train)
tf.summary.scalar("accuracy_validation", accuracy_validation)
tf.summary.scalar("Cross_Entropy_validation", cross_entropy_validation)

# Creating nodes to monitor the distribution of some of the nodes.
tf.summary.histogram("Weights_Hidden_layer", w1)

# Combine the summaries into a single op that generates all the summary data.
merge = tf.summary.merge_all()

# Write this summary data to disk. Directory of the events. If it receives a graph, visualization using tensorboard
# could be provided.
filewriter = tf.summary.FileWriter("lm8_RMSProp")

# A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are
# evaluated. It is important to release these resources when they are no longer required. To do this, either invoke the
#  tf.Session.close method on the session, or use the session as a context manager. (as follows)
# with tf.Session() as sess:
#   sess.run(...)

sess = tf.Session()
filewriter.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# saver.restore(sess=sess,save_path="save/")


for i in range(20000):

    if i % 1000 == 0:

        batch_xs, batch_ys = mnist.validation.next_batch(mini_batch_size)

        # saver.save(sess=sess,save_path="save/")

        accuracy, loss = sess.run((accuracy_validation, cross_entropy_validation),
                                                        feed_dict={x: batch_xs, y_: batch_ys})

        b = (sess.run(merge, feed_dict={x: batch_xs, y_: batch_ys}))


        # print("step %5i accuracy is %g and cross_entropy(loss) is %g" % (i, accuracy, loss))

        # Calculating the validation error

        filewriter.add_summary(b, i)
    else:

        batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)

        # accuracy, loss = sess.run((accuracy_train, cross_entropy), feed_dict={x: batch_xs, y_: batch_ys})

        sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})

        # a = (sess.run(merge, feed_dict={x: batch_xs, y_: batch_ys}))
        #
        # filewriter.add_summary(a, i)



# Calculating the validation error
# batch_xs, batch_ys = mnist.validation.next_batch(mini_batch_size)
# [accuracy_validation] = sess.run([accuracy_train], feed_dict={x: batch_xs, y_: batch_ys})
# print("The accuracy of test:" + str(accuracy_validation))

# Calculating the test error
batch_xs, batch_ys = mnist.test.next_batch(mini_batch_size)
[accuracy_test] = sess.run([accuracy_train], feed_dict={x: batch_xs, y_: batch_ys})
print("The accuracy of test:" + str(accuracy_test))

# To launch tensorboard: tensorboard --logdir=./<folder name you have stored the event files>
# Note to start from the right directory!
# To have multiple diagrams together, add tags to directory as follows:
# tensorboard --logdir=tag_1:./<event_folder_1>,tag_2:./<event_folder_2>,(continue)
# Example: tensorboard --logdir=16:./lm6mb16,64:./lm6mb64
