import numpy as np
import math

class LDA2dGauss():
    def __init__(self):
        self.W = np.array([0])
        self.Wk = np.array([0])
        self.W0 = np.array([0])
        self.mean_arr = np.array([0])
        self.prior_arr = np.array([0])
        self.tot_cov = np.array([0])
        self.num_cls = 0

    def calc_covs(self, X_train, y_train):
        # between class
        SB = np.zeros((X_train.shape[1], X_train.shape[1]))
        tot_m = np.mean(X_train, axis=0)
        tot_m = tot_m.reshape(tot_m.shape[0], 1)
        for cls in range(0, 10):
            X_cls = X_train[y_train == cls]
            N = X_cls.shape[0]
            m = X_cls.mean(axis=0)
            m = m.reshape((m.shape[0], 1))
            SB += N * (m - tot_m).dot((m - tot_m).T)
        # within class
        SW = np.zeros((X_train.shape[1], X_train.shape[1]))
        for cls in range(0, 10):
            temp_SW = np.zeros((X_train.shape[1], X_train.shape[1]))
            X_cls = X_train[y_train == cls]
            m = X_cls.mean(axis=0)
            m = np.reshape(m, (m.shape[0], 1))
            for xn in X_cls:
                xn = np.reshape(xn, (xn.shape[0], 1))
                temp_SW += (xn - m).dot((xn - m).T)
            SW += temp_SW
        return SB, SW

    def project_2D(self, X, y):
        # calculate between/within covariance matrices
        SB, SW = self.calc_covs(X, y)
        # add identity matrix to solve singular matrix problem
        SW += 1e-10 * np.identity(X.shape[1])
        # calculate eigen values/vectors for Sw^-1*SB
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))
        # sort eigen vectors by eigen values
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        # get W
        self.W = eig_vecs[:, :2]
        X_2D = X.dot(self.W)
        return X_2D

    # calc
    # calculate ak for given datapoint
    def calc_ak(self, x, mean_arr, tot_cov, cls_prob_arr, cls, num_features):
        m = mean_arr[cls].reshape((num_features, 1))
        # avoid singular matrix
        # tot_cov += 1e-10 * np.identity(num_features)
        wk = np.linalg.inv(tot_cov).dot(mean_arr[cls])
        wk0 = -0.5 * (m.T.dot(np.linalg.inv(tot_cov))).dot(m) + np.log(cls_prob_arr[cls])
        ak = wk.T.dot(x) + wk0
        return ak

    # fit
    def fit(self, X, y):
        X = self.project_2D(X, y)
        total = X.shape[0]
        num_features = X.shape[1]
        self.num_cls = len(np.unique(y))
        self.prior_arr = np.zeros(self.num_cls)
        self.mean_arr = np.zeros((self.num_cls, num_features))
        self.Wk = np.zeros((self.num_cls, num_features))
        self.W0 = np.zeros(self.num_cls)
        self.tot_cov = np.zeros((num_features, num_features))
        for cls in range(self.num_cls):
            # get X for current cls
            X_cls = X[y == cls]
            # calculate class probs
            count_cls = X_cls.shape[0]
            prior = count_cls / total
            self.prior_arr[cls] = prior
            # calculate mean
            m = np.mean(X_cls, axis=0)
            self.mean_arr[cls] = m.real
            m = m.reshape((m.shape[0], 1))
            # calculate covariance
            cov = np.cov(X_cls.T)
            self.tot_cov += prior*cov.real
        for cls in range (self.num_cls):
            self.Wk[cls] = (np.linalg.inv(self.tot_cov).dot(self.mean_arr[cls])).ravel().real
            self.W0[cls] = (-0.5*(self.mean_arr[cls].T.dot(np.linalg.inv(self.tot_cov))).dot(self.mean_arr[cls]) + np.log(self.prior_arr[cls])).item().real

    def predict(self, X):
        X = X.dot(self.W)
        # predict
        probs_arr = X.dot(self.Wk.T)+self.W0
        res = np.argmax(probs_arr, axis=1)
        return res