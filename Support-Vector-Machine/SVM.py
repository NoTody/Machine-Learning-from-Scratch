import numpy as np
from cvxopt import matrix
from cvxopt import solvers

class SVM(object):
    """
    A class used to generate Support Vector Machine Classifier
    ...

    Attributes
    ----------
    W : 1D numpy array
        weights for SVM classifier
    b : 1D numpy array
        bias for SVM classifier
    C : float
        penalty term of misclassification error
    SV : 1D numpy arrayh
        support vectors for SVM classifier
    kernel : str
        kernel name of SVM (default 'linear')
    gamma : str
        gamma values (default 'auto')
    min_lagrange : float
        minimum lagrange for lagrange multipliers (default 1e-5)

    Methods
    -------
    linear_kernel(self, X, Y)
        linear kernel computation
    rbf_kernel(self, X, Y, gamma=1.0)
        gaussian kernel computation
    compute_kernel(self, X, Y, kernel)
        compute kernel function with given kernel name
    fit(self, X, y)
        fit the dataset to calculate support vectors, weights and bias
    predict_proba(self, X)
        calculate probability output
    predict(self, X)
        get final prediction output by taking sign operator
    """
    def __init__(self, C, kernel='linear', gamma='auto', min_lagrange=1e-5):
        self.W = np.array([])
        self.b = np.array([])
        self.C = (float)(C)
        self.SV = np.array([])
        self.gamma = gamma
        # set min lagrange to choose support vectors
        self.min_lagrange = min_lagrange
        # default kernel(linear)
        self.kernel = self.linear_kernel
        # assign kernel by name
        if kernel == 'linear':
            self.kernel_name = kernel
            self.kernel = self.linear_kernel
        elif kernel == 'rbf':
            self.kernel_name = kernel
            self.kernel = self.rbf_kernel

    # linear kernel computation
    def linear_kernel(self, X, Y):
        return np.inner(X, Y)

    # gaussian kernel computation
    def rbf_kernel(self, X, Y, gamma=1.0):
        gamma = gamma
        X_norm = np.sum(X ** 2, axis=-1)
        Y_norm = np.sum(Y ** 2, axis=-1)
        K = np.exp(-gamma * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)))
        return K

    # compute kernel function with given kernel name
    def compute_kernel(self, X, Y, kernel):
        X_rows, X_cols = X.shape
        Y_rows, Y_cols = Y.shape
        K = np.zeros((X_rows, Y_rows))
        if self.kernel_name == 'rbf':
            K = kernel(X, Y, self.gamma)
        else:
            K = kernel(X, Y)
        return K

    def fit(self, X, y):
        """
        Fit the dataset to calculate support vectors, weights and bias

        Parameters
        ----------
        X : 2D numpy array, independent variables

        y : 1D numpy array, dependent variable
        """
        X_rows, X_cols = X.shape
        # reshape
        y = y.reshape((X_rows, 1))
        if self.kernel_name == 'rbf':
            # compute gamma with sklearn 'scale' fashion (1 / (n_features * X.var()))
            if self.gamma == 'auto':
                self.gamma = 1 / (X_rows * X.var())
            # get gamma from object input
            else:
                self.gamma = (float)(self.gamma)
        # compute kernel for X
        K = self.compute_kernel(X, X, self.kernel)
        GM = np.outer(y, y) * K
        # set parameters for QP solver
        P = matrix(GM)
        q = matrix(-np.ones((X_rows)))
        A = matrix(y.T.astype('float'))
        b = matrix(0.0)
        G = matrix(np.vstack((-np.eye(X_rows), np.eye(X_rows))))
        h = matrix(np.hstack((np.zeros(X_rows), self.C * np.ones(X_rows))))
        # silence solver
        solvers.options['show_progress'] = False
        # run solver
        sol = solvers.qp(P, q, G, h, A, b)
        # store found lagrange multiplier
        lm = np.array(sol['x'])
        # get index of support vector
        idx = (lm > self.min_lagrange).flatten()
        # get support vectors
        self.SV = X[idx]
        # calculate weight(Y*Lambda)
        self.W = (y[idx].flatten() * lm[idx].flatten()).reshape((-1, 1))
        # compute kernel output for Support Vectors
        K = self.compute_kernel(X[idx], X[idx], self.kernel)
        # compute bias
        self.b = np.mean(y[idx] - np.dot(self.W.T, K))

    def predict_proba(self, X):
        """
        Calculate probability output

        Parameters
        ----------
        X : numpy array, independent variables

        Returns
        -------
        probability of each class [ n_samples, n_classes ]
        """
        K = self.compute_kernel(X, self.SV, self.kernel)
        return np.dot(K, self.W) + self.b

    def predict(self, X):
        """
        Get final prediction output by taking sign operator

        Parameters
        ----------
        X : numpy array, independent variables

        Returns
        -------
        class label of each class [ n_samples, n_classes ]
        """
        # compute kernel output for X and Support Vectors
        res = np.sign(self.predict_proba(X))
        return res
