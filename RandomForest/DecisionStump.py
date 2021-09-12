# decision tree classifier
import math
import numpy as np

# entropy
def entropy(y, sample_weights=None):
    y = np.array(y)
    labels, label_counts = np.unique(y, return_counts = True)
    entropy = 0
    if sample_weights is None:
        for i in range(len(labels)):
            probs = label_counts[i] / np.sum(label_counts)
            entropy += -probs * np.log2(probs)
    else:
        # if sample_weights is defined
        for label in labels:
            # get sample weights for current label
            label_weights = sample_weights[y == label]
            # normalize label weights to get probs
            probs = np.sum(label_weights) / np.sum(sample_weights)
            entropy += -probs * np.log2(probs)
    return entropy

# information gain
def information_gain(X, y, feature_idx, threshold, polarity, sample_weights=None):
    X = np.array(X)
    y = np.array(y)
    # Calculate total entropy
    total_entropy = entropy(y, sample_weights=sample_weights)
    # weighted_entropy = (number in current feature / total number) * (entropy of samples belonging to this feature split)
    # if polarity=1, calculate '>'. if polarity=0, calculate '<'.
    if polarity == 1:
        # get classification result based on threshold and polarity
        cls_result = (X[:, feature_idx] >= threshold)
    else:
        cls_result = (X[:, feature_idx] <= threshold)
    # probability(because only one leaf node in decision stump, positive and negative classes include all samples from
    # parent node)
    values, value_counts = np.unique(cls_result, return_counts=True)
    # calculate the weighted entropy
    weighted_entropy = 0
    for i in range(len(values)):
        # probability
        if sample_weights is None:
            probs = value_counts[i] / np.sum(value_counts)
            split_weights = None
        else:
            # calculate weighted prior probability in current split
            #sample_weights_indices = (cls_result == values[i])
            split_weights = sample_weights[cls_result == values[i]]
            probs = np.sum(split_weights) / np.sum(sample_weights)
        # weighted entropy
        weighted_entropy += probs * entropy(y[cls_result == values[i]], sample_weights=split_weights)
    #Calculate the information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

class DecisionStump(object):
    def __init__(self, sample_weights=None):
        self.threshold = None
        self.feature_idx = None
        self.polarity = None
        self.true_label = None
        self.false_label = None
        self.sample_weights = sample_weights

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_rows, X_cols = X.shape
        max_info_gain = -math.inf
        # search for optimal attribute+threshold to split
        for feature_idx in range(X_cols):
            # get dataset of current feature
            X_f = X[:, feature_idx]
            # get all possible thresholds
            thresholds = np.unique(X_f)
            # get intermediate value as thresholds
            thresholds = (thresholds[1:] + thresholds[:-1]) / 2
            for threshold in thresholds:
                # calculate info gain for both bigger than and less than threshold situation
                info_gain_p1 = information_gain(X, y, feature_idx, threshold, 1, sample_weights=self.sample_weights)
                info_gain_p2 = information_gain(X, y, feature_idx, threshold, 0, sample_weights=self.sample_weights)
                # assign polarity based on quantity of the info gain(maximize)
                if info_gain_p1 > info_gain_p2:
                    info_gain = info_gain_p1
                    polarity = 1
                else:
                    info_gain = info_gain_p2
                    polarity = 0
                # update info gain, threshold, feature name and polarity
                #                 print(feature)
                #                print(info_gain)
                if info_gain > max_info_gain:
                    self.threshold = threshold
                    self.feature_idx = feature_idx
                    self.polarity = polarity
                    max_info_gain = info_gain
        # choose split labels
        X_f = X[:, self.feature_idx]
        if self.polarity == 1:
            y_idx = (X_f >= self.threshold)
            labels, labels_count = np.unique(y[y_idx], return_counts=True)
        else:
            y_idx = (X_f <= self.threshold)
            labels, labels_count = np.unique(y[y_idx], return_counts=True)
        # get label count for parent node
        _, tot_labels_count = np.unique(y, return_counts=True)
        # select labeling for split
        # if the first label after split has more samples on leaf, choose as label
        # if none sample weights, count each label as 1, else count each label as
        # their corresponding weights
        if self.sample_weights is None:
            if labels_count[0] > tot_labels_count[0] - labels_count[0]:
                self.true_label = labels[0]
            else:
                self.true_label = -labels[0]
        else:
            # get leaf weights for true and false condition splits
            true_weights = self.sample_weights[y_idx]
            true_label0_weights = true_weights[y[y_idx] == labels[0]]
            true_label1_weights = true_weights[y[y_idx] == labels[1]]

            false_weights = self.sample_weights[~y_idx]
            false_label0_weights = false_weights[y[~y_idx] == labels[0]]
            false_label1_weights = false_weights[y[~y_idx] == labels[1]]
            # assign label
            if np.sum(true_label0_weights) > np.sum(true_label1_weights):
                self.true_label = labels[0]
            else:
                self.true_label = labels[1]

            if np.sum(false_label0_weights) > np.sum(false_label1_weights):
                self.false_label = labels[0]
            else:
                self.false_label = labels[1]

    def predict(self, X):
        X = np.array(X)
        X_rows, X_cols = X.shape
        X_f = X[:, self.feature_idx]
        if self.polarity == 1:
            preds = (X_f >= self.threshold)
        else:
            preds = (X_f <= self.threshold)
        if self.sample_weights is None:
            preds = np.where(preds == True, self.true_label, -self.true_label)
        else:
            preds = np.where(preds == True, self.true_label, self.false_label)
        return preds
