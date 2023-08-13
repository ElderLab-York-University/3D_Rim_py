import numpy as np
import matplotlib.pyplot as plt

# 创建图像和3D坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制球体
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 0.5 * np.outer(np.cos(u), np.sin(v))
y = 0.5 * np.outer(np.sin(u), np.sin(v))
z = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.9)  # alpha 是透明度参数

# 绘制从(0, 0, 1)到(0, 0, -1)的线段
ax.plot([-1, 1], [0, 0], [0, 0], color='blue')

# 绘制点(0, 0, -1)
ax.scatter(1, 0, 0, color='g', s=100)

# 连接(0, 0, -1)到点P(0, 0.5, 0)
ax.plot([0, 1], [0.5, 0], [0, 0], color='r')

# 绘制点P处的法线，从点P到点P+(0, 0.5, 0)
ax.scatter(0, 0, 0.5, color='red', s=100)
ax.quiver(0, 0, 0.5, 0, 0, 0.5, length=0.5, color='purple')

# 设置图像的界限
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# 显示图像
plt.show()
