import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  datasets, layers, optimizers
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print(x_train)
# 创建模型
def create_mode():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense
    ])