import tensorflow as ts

a = ts.constant([[4, 5], [6, 8]])
b = ts.constant([[1, 2], [3, 4]])
c = ts.matmul(a, b)
print(c)

# 张量的使用
# TensorFlow 的张量在概念上类似于多维数组，
# 我们可以使用它来描述数学中的标量（0 维数组）、向量（1 维数组）、矩阵（2 维数组）等各种量
reandom_float = ts.random.uniform(shape=())
zero_vector = ts.zeros(shape=(2))

# 定义两个2×2的常量矩阵
A = ts.constant([[1., 2.], [3., 4.]])
B = ts.constant([[5., 6.], [7., 8.]])

# 计算偏导数
X = ts.constant([[1., 2.], [3., 4.]])
Y = ts.constant([[1.], [2.]])
w = ts.Variable(initial_value=[[1.], [2.]])
# 注意这里的说有数据类型必须都保持一致 4 和 4.0 不是一个类型
b = ts.Variable(initial_value=1.)
with ts.GradientTape() as tape:
    # 范数
    L = 0.5 * ts.reduce_sum(ts.square(ts.matmul(X, w) + b - Y))
w_gard, b_gard = tape.gradient(L, [w, b])  # 计算L(w, b)关于w, b的偏导数

print([L.numpy(), w_gard.numpy(), b_gard.numpy()])


var_a = ts.Variable(3.0)
var_a.assign(var_a + 10.)
print(var_a)

