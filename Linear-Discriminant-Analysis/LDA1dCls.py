import numpy as np

class LDA1dCls(object):
    def __init__(self):
        self.W = np.array([0])
        self.threshold = 0
        self.m1 = np.array([0])
        self.m2 = np.array([0])

    def calc_covs(self, X, y):
        # calculate means
        means = []
        for cls in range(2):
            means.append(np.mean(X[y == cls], axis=0))
        # SB
        m1 = means[0].reshape(X.shape[1],1)
        m2 = means[1].reshape(X.shape[1],1)
        SB = (m2 - m1).dot((m2 - m1).T)
        # SW
        SW = np.zeros((X.shape[1], X.shape[1]))
        X1 = X[y==0]
        X2 = X[y==1]
        for xn in X1:
            xn = xn.reshape(xn.shape[0], 1)
            SW += (xn - m1).dot((xn - m1).T)
        for xn in X2:
            xn = xn.reshape(xn.shape[0], 1)
            SW += (xn - m2).dot((xn - m2).T)
        return SB, SW

    def fit(self, X_train, y_train):
        # calculate between/within covariance matrices
        SB, SW = self.calc_covs(X_train, y_train)
        # calculate eigen values/vectors for Sw^-1*SB
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))
        # sort eigen vectors by eigen values
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        # get W
        self.W = eig_vecs[:, 0]
        # print(self.W)
        # get threshold
        X_1D = X_train.dot(self.W)
        X1 = X_1D[y_train == 0]
        X2 = X_1D[y_train == 1]
        self.threshold = (X1.mean() + X2.mean()) / 2

    def predict(self, X):
        X_1D = X.dot(self.W)
        pred_res = np.empty([X.shape[0], 1])
        if(self.threshold.real>0):
            pred_res = (X_1D < self.threshold).astype(int)
        else:
            pred_res = (X_1D >= self.threshold).astype(int)
        return np.squeeze(pred_res)