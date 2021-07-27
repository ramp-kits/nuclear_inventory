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
        self.model.fit(X, Y)

    def predict(self, X):
        res = self.model.predict(X)
        return res
