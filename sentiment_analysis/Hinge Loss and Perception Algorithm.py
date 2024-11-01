import numpy as np
import random

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def hinge_loss_single(feature_vector, label, theta, theta_0):
    decision_value = label * (np.dot(theta, feature_vector) + theta_0)
    hinge_loss = max(0, 1 - decision_value)
    return hinge_loss

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    total_hinge_loss = np.zeros(len(feature_matrix))

    for i in range(len(feature_matrix)):
        decision_value = labels[i] * (np.dot(theta, feature_matrix[i]) + theta_0)
        total_hinge_loss[i] = max(0, 1 - decision_value)

    return total_hinge_loss.mean()

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    eps = 1e-8
    decision_value = float(label * (np.dot(current_theta, feature_vector) + current_theta_0))
    print(decision_value)

    if abs(decision_value) < eps or decision_value < 0: 
        theta_after_update = current_theta + (label * feature_vector)
        theta_0_after_update = current_theta_0 + label
        return theta_after_update, theta_0_after_update
    
    else:
        return current_theta, current_theta_0

def perceptron(feature_matrix, labels, T):
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            #print(f"Iteration {t}, sample {i}:")
            #print(f"Before update: theta = {theta}, theta_0 = {theta_0}")
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            #print(f"After update: theta = {theta}, theta_0 = {theta_0}\n")
    
    return theta, theta_0

def average_perceptron(feature_matrix, labels, T):
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    theta_sum = 0
    theta_0_sum = 0
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    return theta_sum / (feature_matrix.shape[0] * T), theta_0_sum / (feature_matrix.shape[0] * T)
    #return (1 / (n_samples * T)) * theta_sum, (1 / (n_samples * T)) * theta_0_sum

'''
# Example data
feature_matrix = np.array([
    [0.36412283, 0.11731087, 0.04940689, 0.39959209, 0.07730397],
    [0.364348, 0.38292782, 0.08721637, 0.37820662, 0.16401107],
    [-0.43277533, 0.41948282, 0.25612371, -0.45029175, 0.29633636],
    [0.21907523, 0.21142143, -0.29645209, -0.49917595, 0.30723582],
    [0.46684295, 0.46031002, 0.12740001, -0.49964837, 0.17843848],
    [0.42763864, 0.23233073, 0.09233735, 0.09827444, -0.15160958],
    [0.11751024, -0.3405757, 0.283756, 0.13202176, -0.2305537],
    [-0.25324522, 0.32897978, 0.44996349, 0.19151442, 0.23698451],
    [-0.07026523, -0.18129536, -0.14815642, -0.16168041, 0.4540142],
    [0.21968049, -0.21368761, -0.45401694, 0.18168279, 0.20068676]
])

labels = np.array([-1, 1, -1, 1, 1, -1, 1, -1, 1, 1])
T = 5

theta, theta_0 = perceptron(feature_matrix, labels, T)
print("Final theta:", theta)
print("Final theta_0:", theta_0)

# Correct Output:
#perceptron output is:
#['1.0408847', '0.9286034', '-0.2266164', '0.1812712', '0.4828410']
'''
