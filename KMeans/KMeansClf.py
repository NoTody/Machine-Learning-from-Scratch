import math
import random
import numpy as np

# KMeans classifier
class KMeans(object):
    def __init__(self, num_clusters=3, max_iter=100):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.centroids = np.array([0])

    def loss(self, X, min_idx, centroids):
        # caclualte total loss
        loss = np.linalg.norm(X - centroids[min_idx]) ** 2
        return loss

    def fit(self, X):
        X = np.array(X)
        X_rows, X_cols = X.shape
        # initialize random centroid by randomly sampling from dataset
        idx = random.sample(range(X_rows), self.num_clusters)
        self.centroids = np.array(X[idx, :])
        prev_loss = math.inf
        iter_count = 0
        # stop iteration exceeeds max iteration
        while iter_count < self.max_iter:
            # calcualte distance matrix to store distance
            # of all samples to all centroids
            distances = np.zeros((X_rows, self.num_clusters))
            for centroid_idx in range(self.num_clusters):
                distances[:, centroid_idx] = np.linalg.norm(X - self.centroids[centroid_idx, :], axis=1) ** 2
            # get min distance cluster index for each sample as labels for current samples
            min_idx = np.argmin(distances, axis=1)
            labels = min_idx
            # calculate current loss
            loss = self.loss(X, min_idx, self.centroids)
            # stop if loss decrease less than 1e-3
            if prev_loss - loss < 1e-3:
                break
            prev_loss = loss
            print(f'iteration {iter_count} loss: {loss}')
            # update and reassign centroids
            for centroid_idx in range(self.num_clusters):
                self.centroids[centroid_idx] = np.mean(X[labels == centroid_idx], axis=0)
            iter_count += 1

    def predict(self, X):
        X = np.array(X)
        X_rows, X_cols = X.shape
        # calculate distance matrice
        distances = np.zeros((X_rows, self.num_clusters))
        for centroid_idx in range(self.num_clusters):
            distances[:, centroid_idx] = np.linalg.norm(X - self.centroids[centroid_idx, :], axis=1) ** 2
        # get labels
        min_idx = np.argmin(distances, axis=1)
        preds = min_idx
        return preds