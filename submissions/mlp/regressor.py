import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.model = MLPRegressor(
            solver="adam",
            hidden_layer_sizes=(100, 100, 100),
            max_iter=300,
            batch_size=100,
            random_state=57,
        )

    def fit(self, X, Y):
        self.X_scaling_ = np.max(X, axis=0, keepdims=True)
        self.Y_scaling_ = np.max(Y, axis=0, keepdims=True)
        self.model.fit(X / self.X_scaling_, Y / self.Y_scaling_)

    def predict(self, X):
        res = self.model.predict(X / self.X_scaling_) * self.Y_scaling_
        return res
