# adaboost classifier
import random
import numpy as np
from DecisionStump import DecisionStump

# adaboost classifier
class Adaboost(object):
    def __init__(self, num_iters):
        self.num_iters = num_iters
        self.classifiers = []
        self.classifier_weights = []
        self.sample_weights = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # get number of samples
        X_rows, X_cols = X.shape
        # calculate initial weight
        init_weights = np.ones(X_rows) / X_rows
        self.sample_weights = init_weights
        # iteration loop
        for t in range(self.num_iters):
            # create decision stump with given sample weights
            classifier = DecisionStump(sample_weights=self.sample_weights)
            # train classifier
            classifier.fit(X, y)
            pred = classifier.predict(X)
            # calc epsilon
            weighted_error = self.sample_weights @ (pred != y)
            # calc alpha
            classifier_weight = np.log((1 - weighted_error) / weighted_error) / 2
            # update w_(t+1) for all samples
            self.sample_weights = self.sample_weights * np.exp(-classifier_weight * y * pred)
            # normalize w_(t+1)
            self.sample_weights = self.sample_weights / self.sample_weights.sum()
            # store classifiers and classifier weights
            self.classifiers.append(classifier)
            self.classifier_weights.append(classifier_weight)

    def predict(self, X):
        X = np.array(X)
        classifer_raw_pred = [classifier.predict(X) for classifier in self.classifiers]
        return np.sign(np.asarray(self.classifier_weights).T @ np.asarray(classifer_raw_pred))
