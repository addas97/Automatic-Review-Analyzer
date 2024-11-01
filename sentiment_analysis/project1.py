from string import punctuation, digits
import numpy as np
import random
import matplotlib.pyplot as plt


#==============================================================================
#===  PART I  =================================================================
#==============================================================================

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

    if abs(decision_value) < eps or decision_value < 0: 
        current_theta = current_theta + (label * feature_vector)
        current_theta_0 = current_theta_0 + label
    
    return current_theta, current_theta_0

def perceptron(feature_matrix, labels, T):
    theta = np.zeros(feature_matrix.shape[1]) # Takes number of columns (features) in feature_matrix
    theta_0 = 0.0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            print(f"Iteration {t}, sample {i}:")
            print(f"Before update: theta = {theta}, theta_0 = {theta_0}")
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            print(f"After update: theta = {theta}, theta_0 = {theta_0}\n")
    
    return theta, theta_0

def average_perceptron(feature_matrix, labels, T):
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    theta_sum = np.zeros(feature_matrix.shape[1])
    theta_0_sum = 0
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0
    return (theta_sum / (feature_matrix.shape[0] * T), theta_0_sum / (feature_matrix.shape[0] * T))


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    if label * (np.dot(theta, feature_vector) + theta_0) > 1:
        theta = (1 - L * eta) * theta

    else: 
        theta = ((1 - L * eta) * theta) + (eta * label * feature_vector)
        theta_0 += + eta * label
    
    return theta, theta_0


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    n = 1

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta = 1 / np.sqrt(n)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_0)
            n += 1

    return theta, theta_0


#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    nsample, nfeatures = feature_matrix.shape
    classification = np.zeros(nsample)
    for i in range(nsample):
        if np.dot(feature_matrix[i], theta) + theta_0 > 0:
            classification[i] = 1
        else:
            classification[i] = -1
    return classification

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_predictions = classify(train_feature_matrix, theta, theta_0)
    validation_predictions = classify(val_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, val_labels)
    return (train_accuracy, validation_accuracy) 

def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    text_list = text.split()
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=False):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """

    with open("stopwords.txt", 'r', encoding='utf8') as stoptext:
        stop_words = stoptext.read()
        stop_words = stop_words.replace('\n', " ").split()
    
    indices_by_word = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stop_words: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word

def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word:
                feature_matrix[i, indices_by_word[word]] += 1

    return feature_matrix

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()

# Example Test
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

# theta, theta_0 = perceptron(feature_matrix, labels, T)
# print("Avg. Perceptron")
# print("Final theta:", theta)
# print("Final theta_0:", theta_0)

# theta, theta_0 = average_perceptron(feature_matrix, labels, T)
# print("Avg. Perceptron")
# print("Final theta:", theta)
# print("Final theta_0:", theta_0)

# Correct Output:
#perceptron output is:
#['1.0408847', '0.9286034', '-0.2266164', '0.1812712', '0.4828410']