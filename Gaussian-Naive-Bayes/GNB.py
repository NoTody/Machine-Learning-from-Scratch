import numpy as np
import math
import random

class GNB():
    """
    A class used to generate Gaussian Naive Bayes Classifiers
    ...

    Attributes
    ----------
    m_arr : 1D numpy array
        mean of given features
    v_arr : 1D numpy array
        variance of given features

    Methods
    -------
    fit(self, X, y)
        fit the dataset to find mean and variance of all given features
    gaussian_func(self, x, m, v)
        calculate gaussian distribution for given mean and variance
    predict(self, X)
        get final prediction for Gaussian Naive Bayes
    """
    def __init__(self):
        self.m_arr = np.array([0])
        self.v_arr = np.array([0])
        self.priors = np.array([0])
        self.num_cls = 0

    """
    Calculate and store mean and variance for each class and each feature

    Parameters
    ----------
    X : 2D numpy array, independent variables

    y : 1D numpy array, dependent variable
    """
    def fit(self, X, y):
        total = X.shape[0]
        num_features = X.shape[1]
        self.num_cls = len(np.unique(y))
        self.m_arr = np.zeros((self.num_cls, num_features))
        self.v_arr = np.zeros((self.num_cls, num_features))
        self.priors = np.zeros(self.num_cls)
        for cls in range(self.num_cls):
            # calculate class prob
            X_cls = np.array(X)[y == cls]
            count_cls = X_cls.shape[0]
            prior = count_cls / total
            self.priors[cls] = prior
            for feature_idx in range(num_features):
                # get feature
                cur_feature = np.array(X_cls)[:, feature_idx]
                # calc mean and variance
                m = np.mean(cur_feature)
                v = np.var(cur_feature)
                # assign to cls vs. mean/variance matrix for each feature
                self.m_arr[cls][feature_idx] = m
                self.v_arr[cls][feature_idx] = v

    # calculate gaussian distribution for given mean and variance
    def gaussian_func(self, x, m, v):
        exp = math.exp(-(math.pow(x-m, 2) / (2 * v)))
        return (1 / (math.sqrt(2*math.pi*v))) * exp

    """
    Get final prediction for Gaussian Naive Bayes

    Parameters
    ----------
    X : numpy array, independent variables

    Returns
    -------
    class label of each class [n_samples]
    """
    def predict(self, X):
        result = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            num_features = X.shape[1]
            # posterior probability
            posts = np.zeros(self.num_cls)
            # add epsilon for smoothing variance
            smoothing_ratio = 1e-9
            # calculate posterior for every class
            for cls in range(self.num_cls):
                post = 1
                # adjust variance by adding smoothing_ratio * max_variance
                max_var = np.amax(self.v_arr[cls, :])
                self.v_arr[cls, :] += max_var * smoothing_ratio
                for feature_idx in range(num_features):
                    post *= self.gaussian_func(x[feature_idx], self.m_arr[cls][feature_idx], self.v_arr[cls][feature_idx])
                # store posterior
                posts[cls] = post
            # calculate sum of independent prob
            probs = np.zeros(self.num_cls)
            # get final probs for all classes
            for cls in range(self.num_cls):
                prob = self.priors[cls] * posts[cls]
                probs[cls] = prob
            # assign final pred to result
            result[idx] = np.argmax(probs)
        return result.reshape((result.shape[0], 1))
