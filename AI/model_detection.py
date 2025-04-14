import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.layers import Layer  # type: ignore
import tensorflow.keras.backend as K  # type: ignore
import numpy as np


@tf.keras.utils.register_keras_serializable(package="Custom", name="Attention")
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


class SquatDetectionModel:
    def __init__(
        self,
        scaler_path="GRU/scaler_GRU.pkl",
        model_path="GRU/Squat_detection_GRU.keras",
    ):
        self.scaler = self.load_scaler(scaler_path)
        self.model = self.load_model(model_path)

    def load_scaler(self, path):
        """Load scaler từ file pickle"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_model(self, path):
        """Load mô hình AI"""
        return load_model(path)

    def predict(self, features):
        """Dự đoán tư thế Squat"""
        features = self.scaler.transform(features)
        features = features.reshape(1, -1)
        features = np.expand_dims(features, axis=1)
        return np.argmax(self.model.predict(features))


# Khởi tạo model để dùng trong app
squat_model = SquatDetectionModel()
