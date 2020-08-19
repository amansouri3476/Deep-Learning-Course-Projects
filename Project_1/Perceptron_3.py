import numpy as np
import random
# Details about this code are presented in the report


class Perceptron3(object):

    def __init__(self, no_of_inputs, threshold=1000, learning_rate=0.01, c=0.6):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.c = c
        self.weights = np.array([random.random() for _ in range(no_of_inputs + 1)])
        # self.weights = np.ones(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
                if prediction != label:
                    if summation > self.c:
                        self.weights[1:] -= self.learning_rate * self.c * inputs
                        self.weights[0] -= self.learning_rate * self.c
                    if summation < -self.c:
                        self.weights[1:] -= self.learning_rate * self.c * -1 * inputs
                        self.weights[0] -= self.learning_rate * self.c * -1
                    else:
                        self.weights[1:] -= self.learning_rate * summation * inputs
                        self.weights[0] -= self.learning_rate * summation


