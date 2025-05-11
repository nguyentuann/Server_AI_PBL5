import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.layers import Layer  # type: ignore
import tensorflow.keras.backend as K  # type: ignore
import pickle


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


scaler_path = "GRU/scaler_GRU.pkl"
model_path = "GRU/Squat_detection_GRU.keras"
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

model = load_model(model_path)