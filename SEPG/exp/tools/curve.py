import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义二元二次函数
def quadratic_function(x1, x2):
    return x1**2 + x2**2

# 定义约束条件
def constraint1(x1, x2):
    return x1 - x2 + 1

def constraint2(x1):
    return np.maximum(0, x1)  # x1 >= 0

# 生成 x1 和 x2 的值
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)

# 创建网格点
x1, x2 = np.meshgrid(x1, x2)
y = quadratic_function(x1, x2)

# 应用约束条件
y[constraint1(x1, x2) != 0] = np.nan
y[constraint2(x1) == 0] = np.nan

# 绘制3D曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, y, cmap='viridis' )

# 绘制约束条件可行域
ax.plot_surface(x1, x2, constraint1(x1, x2), color='red', alpha=0.5, linewidth=0, antialiased=False, label='Constraint1')
ax.plot_surface(x1, x2, constraint2(x1), color='blue', alpha=0.5, linewidth=0, antialiased=False, label='Constraint2')

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')

# 显示图例
# ax.legend()

# 显示图形
plt.show()

