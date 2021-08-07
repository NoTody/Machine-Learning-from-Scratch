import copy
import numpy as np

class SVM_ovr(object):
    """
    A class used to generate one versus all Support Vector Machine Classifiers
    ...

    Attributes
    ----------
    estimator : SVM object
        initial estimator to use for training
    num_cls : int
        number of
    all_estimators : list [SVM]
        penalty term of misclassification error

    Methods
    -------
    fit(self, X, y)
        fit the dataset to find all estimators
    predict(self, X)
        get final prediction output by evaluating all trained estimators
    """
    def __init__(self, estimator):
        self.estimator = estimator
        self.num_cls = 0
        self.all_estimators = list()

    def fit(self, X, y):
        """
        Fit the dataset to find all estimators

        Parameters
        ----------
        X : 2D numpy array, independent variables

        y : 1D numpy array, dependent variable
        """
        self.num_cls = len(np.unique(y))
        # fit for num_cls times and predict base on prediction matrix
        for cls in range(self.num_cls):
            cur_estimator = copy.deepcopy(self.estimator)
            # convert multiclass to 1 vs -1
            y_converted = (y == cls).astype(int)
            y_converted[y_converted != 1] = -1
            # fit one SVM with ovr manner
            cur_estimator.fit(X, y_converted)
            self.all_estimators.append(cur_estimator)

    def predict(self, X):
        """
        Get final prediction output by evaluating all trained estimators

        Parameters
        ----------
        X : numpy array, independent variables

        Returns
        -------
        class label of each class [ n_samples, n_classes ]
        """
        # predictions matrix that stores probability of binary classification for each class
        preds_arr = np.zeros((X.shape[0], self.num_cls))
        # fit for num_cls times and predict base on prediction matrix
        for cls in range(self.num_cls):
            result = self.all_estimators[cls].predict_proba(X)
            preds_arr[:, cls] = result.reshape((X.shape[0]))
        final_preds = np.argmax(preds_arr, axis=1).reshape((X.shape[0], 1))
        return final_preds
