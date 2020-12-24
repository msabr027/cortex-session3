import pickle
import numpy as np


model = None
labels = ["setosa", "versicolor", "virginica"]


def init(model_path, metadata):
    global model
    model = pickle.load(open(model_path, "rb"))


def predict(sample, metadata):
    measurements = [
        sample["sepal_length"],
        sample["sepal_width"],
        sample["petal_length"],
        sample["petal_width"],
    ]

    label_id = model.predict(np.array([measurements]))[0]
    return labels[label_id]
