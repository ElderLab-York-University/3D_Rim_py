import numpy as np
from mayavi import mlab
from tvtk.util import ctf
from tvtk.api import tvtk
def f(x, y):
    return -(x**2 + y**2)/10

# 生成网格数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 根据X的值来定义scalar
scalars = np.zeros(X.shape)
scalars[Y > 0] = 1  # x<0 设置为1
scalars[Y < 0] = 0.5  # x<0 设置为1
scalars[Y == 0] = 0

mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
surf = mlab.mesh(X, Y, Z, scalars=scalars, colormap="gray")


# 获取z轴的最小和最大值，确保平面完全覆盖曲面
z_min, z_max = Z.min(), Z.max()
print(str(z_min)+" "+str(z_max))

X_plane_y = X
Y_plane_y = np.zeros_like(Y)
Z_plane_y = np.linspace(Z.min(), Z.max()+1, Y.shape[0]).reshape(-1, 1) * np.ones(Y.shape)
#mlab.mesh(X_plane_y, Y_plane_y, Z_plane_y, color=(0, 0, 1), opacity=0.5)
mlab.mesh(Y_plane_y, X_plane_y, Z_plane_y, color=(1, 0, 0), opacity=0.5)

# 绘制与x=0平面的交线
z_x = f(np.zeros_like(y), y)
mlab.plot3d(np.zeros_like(y), y, z_x, color=(1, 0, 0), tube_radius=0.05)

# 绘制与y=0平面的交线
z_y = f(x, np.zeros_like(x))
mlab.plot3d(x, np.zeros_like(x), z_y, color=(0, 0, 1), tube_radius=0.05)

# 高亮交点
mlab.points3d([0], [0], [0], color=(1, 1, 0), scale_factor=0.3)

arrow_length = 2  # 箭头长度
arrow_tip = f(0, 0) + arrow_length
mlab.quiver3d(0, 0, f(0, 0), 0, 0, arrow_length, mode='arrow', color=(0, 1, 0), scale_factor=0.5)

mlab.show()

