import numpy as np
import random
# Details about this code are presented in the report
class Perceptron4(object):

    def __init__(self, no_of_inputs, threshold=1000, learning_rate=0.01, beta=0.5, alpha=1.13838):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.beta = beta
        self.alpha = alpha
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
                    if summation > self.alpha:
                        self.weights[1:] -= self.learning_rate * self.beta * inputs
                        self.weights[0] -= self.learning_rate * self.beta
                    if summation < -self.alpha:
                        self.weights[1:] -= self.learning_rate * self.beta * -1 * inputs
                        self.weights[0] -= self.learning_rate * self.beta * -1
                    else:
                        self.weights[1:] -= self.learning_rate * summation * inputs * np.exp(-(summation ** 2)) *\
                                            np.exp(self.alpha ** 2) * self.beta / self.alpha
                        self.weights[0] -= self.learning_rate * summation * np.exp(-(summation ** 2)) *\
                                            np.exp(self.alpha ** 2) * self.beta / self.alpha

