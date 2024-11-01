from string import punctuation, digits
import numpy as np
import random

# Part I


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


#pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    return max(0, 1 - label*(theta.T.dot(feature_vector) + theta_0))
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Compute hinge loss and compare with 0
    h_loss = 1 - labels*(np.dot(theta, feature_matrix.T) + theta_0)
    h_loss[h_loss < 0] = 0
    
    return h_loss.mean()
#pragma: coderesponse end


#pragma: coderesponse template
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    
    # Tolerance for floating point errors
    eps = 1e-8
    
    agreement = float(label*(current_theta.dot(feature_vector) + current_theta_0))
    
    if abs(agreement) < eps or agreement < 0:   # 1st condition to check if = 0
            current_theta = current_theta + label*feature_vector
            current_theta_0 = current_theta_0 + label
            
    return (current_theta, current_theta_0)
    
#pragma: coderesponse end


#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0.0
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            print(f"Iteration {t}, sample {i}:")
            current_theta, current_theta_0 = \
            perceptron_single_step_update(feature_matrix[i,:], labels[i], \
                                          current_theta, current_theta_0)
            
    return (current_theta, current_theta_0)

# Example data
feature_matrix = np.array([
    [ 0.48532409, -0.0567855,  0.36824667, -0.28716001,  0.30662714,  0.00963924,  0.04549372, -0.28117785,  0.29127445,  0.31199232],
    [-0.37998421,  0.04291054, -0.34166718, -0.01529387, -0.16885415,  0.22483906,  0.13836197, -0.11557934, -0.18310175, -0.11523374],
    [ 0.25790588, -0.36451603, -0.42368132, -0.33556756,  0.16594336,  0.45891471, -0.25699971, -0.45988629,  0.04790989,  0.0006402 ],
    [ 0.39384064, -0.1233657, -0.15271268, -0.35228015,  0.26561346, -0.26651228,  0.28837639,  0.40292894, -0.48688084, -0.39504014],
    [ 0.10042028, -0.38806312, -0.1349444,  0.33485881, -0.04860299,  0.31963992,  0.07134731, -0.132888, -0.1557277, -0.23399431]
])

labels = np.array([-1, 1, -1, 1, 1])
T = 5

theta, theta_0 = perceptron(feature_matrix, labels, T)
print("Final theta:", theta)
print("Final theta_0:", theta_0)


# Correct Output:
'''
perceptron output is:
['-0.1574856', '-0.0235471', '0.2887369', '0.6704264', '-0.2145464', '-0.1392748', '0.3283470', '0.3269983', '-0.2036376', '-0.2346345']
'''
