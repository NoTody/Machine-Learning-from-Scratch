import numpy as np
import random

class LR():
    """
    A class used to generate Logistic Regression Classifier
    ...

    Attributes
    ----------
    weights_j : 1D numpy array
        weights for Logistic Regression Classifier
    weights_o : float
        bias for Logistic Regression Classifier

    Methods
    -------
    fit(self, X, y)
        fit the dataset to iteratively update weights and bias
    predict_proba(self, X)
        calculate probability output
    predict(self, X)
        get final labels by probability threshold
    """
    def __init__(self):
        # wj
        self.weights_j =np.array([0])
        # w0
        self.weights_o = 0

    # sigmoid function calculation
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    # calculate gradients of objective
    def calc_gradients(self, X, z, y):
        dwj = np.dot(X.T, (z-y)) / y.shape[0]
        dw0 = np.sum(z-y) / y.shape[0]
        return dwj, dw0

    # weight updating
    def update_weight(self, weight, learning_rate, gradient):
        return weight - learning_rate * gradient

    def fit(self, X, y):
        """
        Fit the dataset to iteratively update weights and bias

        Parameters
        ----------
        X : 2D numpy array, independent variables

        y : 1D numpy array, dependent variable
        """
        X = X.astype(float)
        y = y.reshape((y.shape[0],1))
        X_rows, X_cols=X.shape
        # weights initialization
        self.weights_j = np.full((X_cols, 1), random.uniform(-0.01,0.01))
        self.weights_o = random.uniform(-0.01,0.01)
        # set max iterations
        maxIteration = 1000
        step_size=3e-5
        prev_cost = 0
        # create matrix to store previous gradients
        for t in range(maxIteration):
            # calc gradients
            z = self.sigmoid(X.dot(self.weights_j)+self.weights_o)
            dwj, dw0 = self.calc_gradients(X, z, y)
            # update weights
            self.weights_j = self.update_weight(self.weights_j, step_size, dwj)
            self.weights_o = self.update_weight(self.weights_o, step_size, dw0)
            cost = (-y*np.log(z)-(1-y)*np.log(1-z)).mean()
            if cost < 0.01:
                break

    def predict_proba(self, X):
        """
        Calculate probability output

        Parameters
        ----------
        X : numpy array, independent variables

        Returns
        -------
        probability of each datapoint [n_samples]
        """
        return self.sigmoid(X.dot(self.weights_j)+self.weights_o)

    def predict(self, X):
        """
        Get final labels by probability threshold

        Parameters
        ----------
        X : numpy array, independent variables

        Returns
        -------
        binary class label of each datapoint [n_samples]
        """
        res = (self.predict_proba(X) >= 0.5).astype(int)
        return res