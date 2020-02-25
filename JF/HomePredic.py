import tensorflow as ts
import numpy as np

x_raw = np.array([2013,2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
x = (x_raw - x_raw.min())/(x_raw.max() - x_raw.min())
y = (y_raw - y_raw.min())/(y_raw.max() - y_raw.min())
print(x)
print(y)
a, b = 0, 0
num_epoch = 10
learning_rate = 1e-3
# 0.001
print(learning_rate)
for e in range(num_epoch):
    y_pred = a*x + b
    grad_a, grad_b = (y_pred - y).dot(x), (y_pred - y).sum()
    print("grad_a = {}".format(grad_a))
    print("grad_b = {}".format(grad_b))
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
print(a, b)
'''
经常需要手工求函数关于参数的偏导数。
如果是简单的函数或许还好，但一旦函数的形式变得复杂（尤其是深度学习模型），手工求导的过程将变得非常痛苦，甚至不可行。
经常需要手工根据求导的结果更新参数。
这里使用了最基础的梯度下降方法，因此参数的更新还较为容易。但如果使用更加复杂的参数更新方法（例如 Adam 或者 Adagrad），这个更新过程的编写同样会非常繁杂。
'''

# TensorFlow 下的线性回归
tx = ts.constant(x)
ty = ts.constant(y)
ta = ts.Variable(initial_value=0.0)
tb = ts.Variable(initial_value=0.0)
variables = [ta, tb]
tnum_epoch = 1000
optimizer = ts.optimizers.SGD(learning_rate=1e-3)
for e in range(tnum_epoch):
    with ts.GradientTape() as tape:
        ty_pred = ta*tx + tb
        loss = 0.5 * ts.reduce_sum(ts.square(ty_pred - ty))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
print("{a} {b}".format(ta, tb))
