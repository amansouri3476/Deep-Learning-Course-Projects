import numpy as np
import matplotlib.pyplot as plt
import Rastrigin
import Levi
import Ackley
import random
# import Bukin
# import N_D_Rastrigin


############################################## Gradient Descent ##############################################

# Hyper parameter: just learning rate (eta)

# noinspection PyUnusedLocal
def gradient_descent(target_function, coordinates, gradient_of_target_function, optimization_hyper_parameters):

    value = []
    counter = 0

    learning_rate_gd = optimization_hyper_parameters

    gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                 gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2)

    while gradient_magnitude > 1e-4 and counter < 1e4:

        [x_new, y_new] = list(np.array(coordinates) - learning_rate_gd *
                              gradient_of_target_function(coordinates[0], coordinates[1]))

        coordinates = [x_new, y_new]

        gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                     gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2)

        value.append(target_function(x_new, y_new))
        counter += 1

    return value


############################################## Nesterov ##############################################


# Hyper parameters: learning rate (eta) and momentum (gamma)

def nesterov(target_function, coordinates, gradient_of_target_function, optimization_hyper_parameters):

    value = []
    counter = 0

    learning_rate_nesterov = optimization_hyper_parameters[0]
    momentum_coeff = optimization_hyper_parameters[1]
    gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                 gradient_of_target_function(coordinates[0], coordinates[1])[1] ** 2)
    # First Iteration of this method has to be done using regular GD since we do not have any previous iteration.
    [x_new, y_new] = list(np.array(coordinates) - learning_rate_nesterov *
                          gradient_of_target_function(coordinates[0], coordinates[1]))
    coordinates = [x_new, y_new]
    last_iteration_gradient = gradient_of_target_function(coordinates[0], coordinates[1])

    while counter < 1e4 and gradient_magnitude > 1e-4:

        update_term = momentum_coeff * last_iteration_gradient + \
                      learning_rate_nesterov * gradient_of_target_function(coordinates[0] -
                                                                           momentum_coeff * last_iteration_gradient[0],
                                                                           coordinates[1] - momentum_coeff *
                                                                           last_iteration_gradient[1])
        [x_new, y_new] = list(np.array(coordinates) - update_term)
        coordinates = [x_new, y_new]
        last_iteration_gradient = update_term

        gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                     gradient_of_target_function(coordinates[0], coordinates[1])[1] ** 2)

        value.append(target_function(x_new, y_new))
        counter += 1

    return value


############################################## RMSProp ##############################################


# Hyper parameters: learning rate (eta) and epsilon (gamma)

def rms_prop(target_function, coordinates, gradient_of_target_function, optimization_hyper_parameters):

    value = []
    counter = 0

    learning_rate_rms_prop = optimization_hyper_parameters[0]
    epsilon = optimization_hyper_parameters[1]
    gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                 gradient_of_target_function(coordinates[0], coordinates[1])[1] ** 2)

    # First Iteration of this method has to be done using regular GD since we do not have any previous iteration.
    [x_new, y_new] = list(np.array(coordinates) - learning_rate_rms_prop *
                          gradient_of_target_function(coordinates[0], coordinates[1]))

    coordinates = [x_new, y_new]
    decaying_average_of_squared_gradients = 0
    n = 0
    decaying_average_of_squared_gradients += gradient_magnitude ** 2
    n += 1

    while counter < 1e4 and gradient_magnitude > 1e-4:
        # print(gradient_of_target_function(coordinates[0], coordinates[1]))
        n += 1
        average_of_squared_gradients = (0.9 * decaying_average_of_squared_gradients + 0.1 * gradient_magnitude ** 2) / n
        update_term = (learning_rate_rms_prop / np.sqrt(average_of_squared_gradients + epsilon)) * \
                      gradient_of_target_function(coordinates[0], coordinates[1])
        [x_new, y_new] = list(np.array(coordinates) - update_term)
        coordinates = [x_new, y_new]
        gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                     gradient_of_target_function(coordinates[0], coordinates[1])[1] ** 2)

        value.append(target_function(x_new, y_new))
        counter += 1

    return value

############################################## Adam ##############################################


# Hyper parameters: learning rate (eta), first momentum of gradient coefficient (beta_1),
# second momentum of gradient coefficient (beta_2), constant epsilon to avoid denominator from vanishing.

def adam(target_function, coordinates, gradient_of_target_function, optimization_hyper_parameters):

    value = []
    counter = 0

    learning_rate_adam = optimization_hyper_parameters[0]
    beta_1 = optimization_hyper_parameters[1]
    beta_2 = optimization_hyper_parameters[2]
    epsilon = optimization_hyper_parameters[3]
    gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                 gradient_of_target_function(coordinates[0], coordinates[1])[1] ** 2)
    # First Iteration of this method has to be done using regular GD since we do not have any previous iteration.
    [x_new, y_new] = list(np.array(coordinates) - learning_rate_adam *
                          gradient_of_target_function(coordinates[0], coordinates[1]))
    coordinates = [x_new, y_new]
    decaying_average_of_squared_gradients = 0
    decaying_average_of_gradients = 0
    n = 0
    decaying_average_of_squared_gradients += gradient_magnitude ** 2
    decaying_average_of_gradients += gradient_of_target_function(coordinates[0], coordinates[1])
    n += 1

    while counter < 1e4 and gradient_magnitude > 1e-4:
        n += 1
        average_of_squared_gradients = (beta_2 * decaying_average_of_squared_gradients + (1 - beta_2) *
                                        gradient_magnitude ** 2) / n
        average_of_squared_gradients_bias_corrected = average_of_squared_gradients / (1 - beta_2 ** n)
        average_of_gradients = (beta_1 * decaying_average_of_gradients + (1 - beta_1) *
                                gradient_of_target_function(coordinates[0], coordinates[1])) / n
        average_of_gradients_bias_corrected = average_of_gradients / (1 - beta_1 ** n)

        update_term = (learning_rate_adam / (np.sqrt(average_of_squared_gradients_bias_corrected) + epsilon)) * \
                      average_of_gradients_bias_corrected

        [x_new, y_new] = list(np.array(coordinates) - update_term)
        coordinates = [x_new, y_new]

        gradient_magnitude = np.sqrt(gradient_of_target_function(coordinates[0], coordinates[1])[0] ** 2 +
                                     gradient_of_target_function(coordinates[0], coordinates[1])[1] ** 2)

        value.append(target_function(x_new, y_new))
        counter += 1

    return value

############################################## Newton's Method ##############################################


def newton_method(target_function, coordinates, gradient_of_target_function, hessian_of_target_function,
                  optimization_hyper_parameters):
    counter = 0
    value = []
    step_size = optimization_hyper_parameters

    # while counter < 1e4 and target_function(coordinates[0], coordinates[1]) > 1e-3:
    while target_function(coordinates[0], coordinates[1]) > 1e-4 and counter < 1e4:

        hessian = hessian_of_target_function(coordinates[0], coordinates[1])
        hessian_inv = np.linalg.inv(hessian)
        [x_new, y_new] = list(np.array(coordinates) - step_size * np.matmul(hessian_inv, gradient_of_target_function(
            coordinates[0], coordinates[1])))
        counter += 1
        coordinates = [x_new, y_new]
        value.append(target_function(coordinates[0], coordinates[1]))

    return value


############################################# TEST FUNCTIONS #############################################


############################# Rastrigin #############################

# x = 2 * 5.12 * random.random() - 5.12
# y = 2 * 5.12 * random.random() - 5.12
#
# fig = plt.figure()
# ax = plt.subplot(111)
# value_1 = gradient_descent(Rastrigin.rastrigin, [x, y], Rastrigin.rastrigin_gradient, 0.0001)
# ax.plot(value_1, label='GD-0.0001')
# plt.show()
#
# plt.hold(True)
#
# value_2 = nesterov(Rastrigin.rastrigin, [x, y], Rastrigin.rastrigin_gradient, [0.001, 0.9])
# ax.plot(value_2, label='Nesterov')
# plt.show()
#
# plt.hold(True)
#
# value_3 = rms_prop(Rastrigin.rastrigin, [x, y], Rastrigin.rastrigin_gradient, [0.001, 1e-8])
# ax.plot(value_3, label='RMSProp')
# plt.show()
#
# plt.hold(True)
#
# value_4 = adam(Rastrigin.rastrigin, [x, y], Rastrigin.rastrigin_gradient, [0.001, 0.9, 0.999, 1e-8])
# ax.plot(value_4, label='Adam')
# plt.show()
# plt.hold(True)
#
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
# plt.xlim([-10, 500])
#
############################# Ackley #############################

# x = 2 * 5 * random.random() - 5
# y = 2 * 5 * random.random() - 5
# #
# fig_1 = plt.figure()
# ax = plt.subplot(111)
# value_1 = gradient_descent(Ackley.ackley, [x, y], Ackley.ackley_gradient, 0.001)
# ax.plot(value_1, label='GD')
# plt.show()
#
# plt.hold(True)
#
#
# value_2 = nesterov(Ackley.ackley, [x, y], Ackley.ackley_gradient, [0.001, 0.9])
# ax.plot(value_2, label='Nesterov')
# plt.show()
#
# plt.hold(True)
#
# value_3 = rms_prop(Ackley.ackley, [x, y], Ackley.ackley_gradient, [0.0001, 1e-8])
# ax.plot(value_3, label='RMSProp')
# plt.show()
#
# plt.hold(True)
#
# value_4 = adam(Ackley.ackley, [x, y], Ackley.ackley_gradient, [0.0001, 0.9, 0.999, 1e-8])
# ax.plot(value_4, label='Adam-0.0001')
# plt.show()
#
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
# plt.xlim([-10, 500])

############################# Levi #############################

# x = 2 * 10 * random.random() - 10
# y = 2 * 10 * random.random() - 10
#
# fig_2 = plt.figure()
# ax = plt.subplot(111)
# value_1 = gradient_descent(Levi.levi, [x, y], Levi.levi_gradient, 0.0001)
# ax.plot(value_1, label='GD-0.0001')
# plt.show()
#
# plt.hold(True)
#
# value_2 = nesterov(Levi.levi, [x, y], Levi.levi_gradient, [0.001, 0.9])
# ax.plot(value_2, label='Nesterov')
# plt.show()
#
# plt.hold(True)
#
#
# value_3 = rms_prop(Levi.levi, [x, y], Levi.levi_gradient, [0.0001, 1e-8])
# ax.plot(value_3, label='RMSProp-0.0001')
# plt.show()
#
# plt.hold(True)
#
# value_4 = adam(Levi.levi, [x, y], Levi.levi_gradient, [0.0001, 0.9, 0.999, 1e-8])
# ax.plot(value_4, label='Adam-0.0001')
# plt.show()
#
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
# plt.xlim([-10, 500])

##################################### Testing Newton Method #####################################

# x = 2 * 5.12 * random.random() - 5.12
# y = 2 * 5.12 * random.random() - 5.12
#
# value = newton_method(Rastrigin.rastrigin, [x, y], Rastrigin.rastrigin_gradient, Rastrigin.rastrigin_hessian, 1)
# plt.plot(value)
# plt.show()
#
#
# x = 2 * 10 * random.random() - 10
# y = 2 * 10 * random.random() - 10
#
# value = newton_method(Levi.levi, [x, y], Levi.levi_gradient, Levi.levi_hessian, 1)
# plt.plot(value)
# plt.show()

