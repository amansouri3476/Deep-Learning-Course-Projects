import pandas as pd
# import random
from Perceptron_1 import Perceptron1
from Perceptron_2 import Perceptron2
from Perceptron_3 import Perceptron3
from Perceptron_4 import Perceptron4

import numpy as np
import matplotlib.pyplot as plt

# Load data
file_name = "digits.csv"
data = pd.read_csv(file_name, sep=",", header=0)
first_digit = 1
second_digit = 4

# Finding indexes of data which have labels equal to first_digit
indexes_of_first_digit = []
for i in range(len(data)):
    if data.loc[i].label == first_digit:
        indexes_of_first_digit.append(i)
first_digit_train_indexes = indexes_of_first_digit[0:int(np.floor(len(indexes_of_first_digit) * 0.8))]
first_digit_test_indexes = indexes_of_first_digit[int(np.floor(len(indexes_of_first_digit) * 0.8)) + 1:]

# Finding indexes of data which have labels equal to second_digit
indexes_of_second_digit = []
for i in range(len(data)):
    if data.loc[i].label == second_digit:
        indexes_of_second_digit.append(i)
second_digit_train_indexes = indexes_of_second_digit[0:int(np.floor(len(indexes_of_second_digit) * 0.8))]
second_digit_test_indexes = indexes_of_second_digit[int(np.floor(len(indexes_of_second_digit) * 0.8)) + 1:]


train_indexes = first_digit_train_indexes + second_digit_train_indexes
test_indexes = first_digit_test_indexes + second_digit_test_indexes

labels = []

# Mapping from first_digit and second_digit to 0 and 1
for i in train_indexes:
    if data.loc[i].label == 1:
        labels.append(1)
    else:
        labels.append(0)

training_inputs = []

# training data is prepared to be used in perceptron.train according to Perceptron object defined in perceptron.
for i in train_indexes:
    training_inputs.append(np.array([data.loc[i].feature_1, data.loc[i].feature_2, data.loc[i].feature_3,
                                     data.loc[i].feature_4, data.loc[i].feature_5, data.loc[i].feature_6,
                                     data.loc[i].feature_7, data.loc[i].feature_8, data.loc[i].feature_9,
                                     data.loc[i].feature_10, data.loc[i].feature_11, data.loc[i].feature_12,
                                     data.loc[i].feature_13, data.loc[i].feature_14, data.loc[i].feature_15,
                                     data.loc[i].feature_16]))

# Our perceptron has 16 distinct features and its other parameters are set to default value.
# Four perceptron objects according to four models of different loss function
perceptron_1 = Perceptron1(16)
perceptron_2 = Perceptron2(16)
perceptron_3 = Perceptron3(16)
perceptron_4 = Perceptron4(16)

# Perceptrons are trained using transformed to binary labels and merged features.
perceptron_1.train(training_inputs, np.array(labels))
perceptron_2.train(training_inputs, np.array(labels))
perceptron_3.train(training_inputs, np.array(labels))
perceptron_4.train(training_inputs, np.array(labels))

test_inputs = []
test_labels = []
for i in test_indexes:
    test_inputs.append(np.array([data.loc[i].feature_1, data.loc[i].feature_2, data.loc[i].feature_3,
                                 data.loc[i].feature_4, data.loc[i].feature_5, data.loc[i].feature_6,
                                 data.loc[i].feature_7, data.loc[i].feature_8, data.loc[i].feature_9,
                                 data.loc[i].feature_10, data.loc[i].feature_11, data.loc[i].feature_12,
                                 data.loc[i].feature_13, data.loc[i].feature_14, data.loc[i].feature_15,
                                 data.loc[i].feature_16]))
    if data.loc[i].label == first_digit:
        test_labels.append(1)
    else:
        test_labels.append(0)

result_1 = []
result_2 = []
result_3 = []
result_4 = []
for i in range(len(test_indexes)):
    result_1.append(perceptron_1.predict(test_inputs[i]))
    result_2.append(perceptron_2.predict(test_inputs[i]))
    result_3.append(perceptron_3.predict(test_inputs[i]))
    result_4.append(perceptron_4.predict(test_inputs[i]))

# Calculating the scores for all models
counter_1 = 0
counter_2 = 0
counter_3 = 0
counter_4 = 0

for i in range(len(test_indexes)):
    if result_1[i] == test_labels[i]:
        counter_1 += 1
    if result_2[i] == test_labels[i]:
        counter_2 += 1
    if result_3[i] == test_labels[i]:
        counter_3 += 1
    if result_4[i] == test_labels[i]:
        counter_4 += 1

score_1 = counter_1/len(test_indexes) * 100
score_2 = counter_2/len(test_indexes) * 100
score_3 = counter_3/len(test_indexes) * 100
score_4 = counter_4/len(test_indexes) * 100

print('Precision of Model #1 is:')
print(score_1)

print('Precision of Model #2 is:')
print(score_2)

print('Precision of Model #3 is:')
print(score_3)

print('Precision of Model #4 is:')
print(score_4)

# Calculating Confusion Matrices for all models

True_Positive_counter_1 = 0
False_Positive_counter_1 = 0
False_Negative_counter_1 = 0
True_Negative_counter_1 = 0

True_Positive_counter_2 = 0
False_Positive_counter_2 = 0
False_Negative_counter_2 = 0
True_Negative_counter_2 = 0

True_Positive_counter_3 = 0
False_Positive_counter_3 = 0
False_Negative_counter_3 = 0
True_Negative_counter_3 = 0

True_Positive_counter_4 = 0
False_Positive_counter_4 = 0
False_Negative_counter_4 = 0
True_Negative_counter_4 = 0

for i in range(len(test_indexes)):
    if result_1[i] == 1 and test_labels[i] == 1:  # True Positive
        True_Positive_counter_1 += 1
    elif result_1[i] == 1 and test_labels[i] == 0:  # False Positive
        False_Positive_counter_1 += 1
    elif result_1[i] == 0 and test_labels[i] == 1:  # False Negative
        False_Negative_counter_1 += 1
    elif result_1[i] == 0 and test_labels[i] == 0:  # True Negative
        True_Negative_counter_1 += 1

    if result_2[i] == 1 and test_labels[i] == 1:  # True Positive
        True_Positive_counter_2 += 1
    elif result_2[i] == 1 and test_labels[i] == 0:  # False Positive
        False_Positive_counter_2 += 1
    elif result_2[i] == 0 and test_labels[i] == 1:  # False Negative
        False_Negative_counter_2 += 1
    elif result_2[i] == 0 and test_labels[i] == 0:  # True Negative
        True_Negative_counter_2 += 1

    if result_3[i] == 1 and test_labels[i] == 1:  # True Positive
        True_Positive_counter_3 += 1
    elif result_3[i] == 1 and test_labels[i] == 0:  # False Positive
        False_Positive_counter_3 += 1
    elif result_3[i] == 0 and test_labels[i] == 1:  # False Negative
        False_Negative_counter_3 += 1
    elif result_3[i] == 0 and test_labels[i] == 0:  # True Negative
        True_Negative_counter_3 += 1

    if result_4[i] == 1 and test_labels[i] == 1:  # True Positive
        True_Positive_counter_4 += 1
    elif result_4[i] == 1 and test_labels[i] == 0:  # False Positive
        False_Positive_counter_4 += 1
    elif result_4[i] == 0 and test_labels[i] == 1:  # False Negative
        False_Negative_counter_4 += 1
    elif result_4[i] == 0 and test_labels[i] == 0:  # True Negative
        True_Negative_counter_4 += 1


Confusion_Matrix_1 = [[True_Positive_counter_1, False_Negative_counter_1],
                      [False_Positive_counter_1, True_Negative_counter_1]]

Confusion_Matrix_2 = [[True_Positive_counter_2, False_Negative_counter_2],
                      [False_Positive_counter_2, True_Negative_counter_2]]

Confusion_Matrix_3 = [[True_Positive_counter_3, False_Negative_counter_3],
                      [False_Positive_counter_3, True_Negative_counter_3]]

Confusion_Matrix_4 = [[True_Positive_counter_4, False_Negative_counter_4],
                      [False_Positive_counter_4, True_Negative_counter_4]]

print('Confusion Matrix Model #1')
print(Confusion_Matrix_1)

print('Confusion Matrix Model #2')
print(Confusion_Matrix_2)

print('Confusion Matrix Model #3')
print(Confusion_Matrix_3)

print('Confusion Matrix Model #4')
print(Confusion_Matrix_4)

# Plotting the results
plt.subplot(2, 2, 1)
plt.title('Perceptron #1', color='blue')
plt.plot(result_1, '.')
plt.subplot(2, 2, 2)
plt.title('Perceptron #2', color='blue')
plt.plot(result_2, '.')
plt.subplot(2, 2, 3)
plt.title('Perceptron #3', color='blue')
plt.plot(result_3, '.')
plt.subplot(2, 2, 4)
plt.title('Perceptron #4', color='blue')
plt.plot(result_4, '.')
plt.show()

plt.title('Perceptron #1', color='blue')
plt.plot(result_1, '.')
plt.show()

plt.title('Perceptron #2', color='blue')
plt.plot(result_2, '.')
plt.show()

plt.title('Perceptron #3', color='blue')
plt.plot(result_3, '.')
plt.show()

plt.title('Perceptron #4', color='blue')
plt.plot(result_4, '.')
plt.show()
