import numpy as np
import matplotlib.pyplot as plt

# 定义边缘密度函数 f_X(λ1) 和 f_Y(λ2)
def f_X(lambda_1):
    if -1 <= lambda_1 < 0:
        return 1 + lambda_1
    elif 0 <= lambda_1 <= 1:
        return 1 - lambda_1
    else:
        return 0

def f_Y(lambda_2):
    if -1 <= lambda_2 < 0:
        return 1 + lambda_2
    elif 0 <= lambda_2 <= 1:
        return 1 - lambda_2
    else:
        return 0

# 绘制边缘密度函数
lambda_1_values = np.linspace(-1, 1, 100)
f_X_values = [f_X(l) for l in lambda_1_values]

lambda_2_values = np.linspace(-1, 1, 100)
f_Y_values = [f_Y(l) for l in lambda_2_values]

plt.figure(figsize=(10, 5))

# 绘制 f_X(λ1)
plt.subplot(1, 2, 1)
plt.plot(lambda_1_values, f_X_values, label='f_X(λ1)')
plt.title('Marginal Density f_X(λ1)')
plt.xlabel('λ1')
plt.ylabel('f_X(λ1)')
plt.grid(True)

# 绘制 f_Y(λ2)
plt.subplot(1, 2, 2)
plt.plot(lambda_2_values, f_Y_values, label='f_Y(λ2)', color='orange')
plt.title('Marginal Density f_Y(λ2)')
plt.xlabel('λ2')
plt.ylabel('f_Y(λ2)')
plt.grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 创建一个新的图形
plt.figure(figsize=(6, 6))

# 定义三角形区域的顶点
# 第一个三角形 (位于第一象限)
triangle1_x = [0, 1, 0]
triangle1_y = [0, 0, 1]

# 第二个三角形 (位于第三象限)
triangle2_x = [0, -1, 0]
triangle2_y = [0, 0, -1]

# 绘制两个三角形
plt.fill(triangle1_x, triangle1_y, color='lightblue', alpha=0.6, label='First Quadrant Triangle')
plt.fill(triangle2_x, triangle2_y, color='lightgreen', alpha=0.6, label='Third Quadrant Triangle')

# 绘制坐标轴
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)

# 设置坐标轴的范围
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

# 设置刻度标签
plt.xticks(np.arange(-1.5, 2, 0.5))
plt.yticks(np.arange(-1.5, 2, 0.5))

# 添加网格
plt.grid(True)

# 标记直线方程 λ1 + λ2 = 1 和 λ1 + λ2 = -1
plt.plot([-1.5, 1.5], [1.5, -1.5], color='red', linestyle='--', label=r'$\lambda_1 + \lambda_2 = 1$')
plt.plot([-1.5, 1.5], [-1.5, 1.5], color='blue', linestyle='--', label=r'$\lambda_1 + \lambda_2 = -1$')

# 设置标题和标签
plt.title('Joint Probability Density Region for X and Y')
plt.xlabel(r'$\lambda_1$')
plt.ylabel(r'$\lambda_2$')

# 显示图例
plt.legend()

# 展示图像
plt.show()
