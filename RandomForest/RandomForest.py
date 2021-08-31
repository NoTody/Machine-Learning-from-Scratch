# random forest
import random
import numpy as np
from DecisionStump import DecisionStump

class RandomForest(object):
    def __init__(self, num_features, num_trees):
        # num of features to select
        self.num_features = num_features
        # num of trees to select
        self.num_trees = num_trees
        # a list to store created trees
        self.trees = []
        # a list to select selected features for each tree
        self.trees_features = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_rows, X_cols = X.shape
        num_samples = (int)((2 / 3) * X_rows)
        # create trees
        for i in range(self.num_trees):
            # bootstrape sample from training data
            idx = random.sample(range(X_rows), num_samples)
            X_boot = X[idx, :]
            y_boot = y[idx]
            # select random attributes
            selected_features = random.sample(range(X_cols), self.num_features)
            # train classifier
            classifier = DecisionStump()
            classifier.fit(X_boot[:, selected_features], y_boot)
            # store current tree
            self.trees.append(classifier)
            self.trees_features.append(selected_features)

    def predict(self, X):
        X = np.array(X)
        X_rows, X_cols = X.shape
        sum_preds = np.zeros(X_rows)
        for tree, selected_features in zip(self.trees, self.trees_features):
            sum_preds += tree.predict(X[:, selected_features])
        # calculate final prediction
        final_preds = np.where(sum_preds > 0, +1, -1)
        return final_preds