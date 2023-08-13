import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
from scipy.spatial import Delaunay

def f(x, y):
    return -(x**2 + y**2) / 50

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

tri = Delaunay(np.c_[X.ravel(), Y.ravel()]).simplices

# 根据X的值来定义scalar
scalars = np.zeros(X.shape)
scalars[X >= 0] = 1

# 自定义颜色映射
colors = np.array([(150, 150, 150, 255),   # 深灰色
                   (220, 220, 220, 255)])  # 浅灰色

# 创建LUT
lut = tvtk.LookupTable()
lut.table = colors

mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

# 使用triangular_mesh绘制三角形化的曲面
triangular_mesh = mlab.triangular_mesh(X.ravel(), Y.ravel(), Z.ravel(), tri, scalars=scalars.ravel(), colormap="gray")
triangular_mesh.module_manager.scalar_lut_manager.lut = lut

# 添加交线和箭头
z_plane = (Z.max() + Z.min()) / 2
X_plane, Y_plane = np.meshgrid(x, y)

mlab.mesh(X_plane, Y_plane, np.full_like(X_plane, z_plane), color=(0.5, 0.5, 0.5), opacity=0.3)
mlab.mesh(X_plane, np.full_like(Y_plane, 0), Z, color=(0.5, 0.5, 0.5), opacity=0.3)
mlab.mesh(np.full_like(X_plane, 0), Y_plane, Z, color=(0.5, 0.5, 0.5), opacity=0.3)
mlab.points3d([0], [0], [f(0, 0)], color=(0, 0, 1), scale_factor=0.3)
mlab.quiver3d(0, 0, f(0, 0), 0, 0, 1, mode='arrow', color=(0, 0, 0), scale_factor=1)

mlab.show()
