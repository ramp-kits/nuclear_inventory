import os, pickle, string
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.utils import shuffle
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import ShuffleSplit

problem_title = "Isotopic inventory of a nuclear reactor core in operation"

_target_names = ["Y_" + j for j in list(string.ascii_uppercase)]

Predictions = rw.prediction_types.make_regression(label_names=_target_names)
workflow = rw.workflows.Regressor()


class MSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="MSE", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mse = (np.square(y_true - y_pred)).mean()
        return mse


class MAPE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="MAPE", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mape = (np.abs(y_true - y_pred) / y_true).mean()
        return mape


score_types = [
    MSE(name="MSE"),
    MAPE(name="MAPE"),
]


def _get_data(path=".", split="train"):
    # load pre-prepared dataset aggregating all of the different input data
    # ( for the training dataset, these are composed of 920 different
    # simulations of an operating reactor )
    dataset = pickle.load(
        open(os.path.join(path, "data", f"{split}_data_python3.pickle"), "rb")
    )

    # Isotopes are named from A to Z
    alphabet = list(string.ascii_uppercase)

    # At T=0 only isotopes from A to H are != 0. Those are the input composition
    # The input parameter space is composed of those initial compositions + operating parameters p1 to p5
    input_params = alphabet[:8] + ["p1", "p2", "p3", "p4", "p5"]

    data = dataset[alphabet].add_prefix("Y_")
    data["times"] = dataset["times"]
    data = data[data["times"] > 0.0]

    temp = pd.DataFrame(
        np.repeat(dataset.loc[0][input_params].values, 80, axis=0),
        columns=input_params
    ).reset_index(drop=True)
    data = pd.concat([temp, data.reset_index(drop=True)], axis=1)

    data = shuffle(data, random_state=57)

    X = data[input_params + ["times"]].to_numpy()
    Y = data[["Y_" + j for j in alphabet]].to_numpy()
    return X, Y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, random_state=57)
    return cv.split(X, y)
