import matplotlib.pyplot as plt
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def perspective_matrix(fov, aspect, near, far):
    # fov 应该是以度为单位，将其转换为弧度
    fov_rad = np.deg2rad(fov)

    # 计算透视投影矩阵的元素
    f = 1 / np.tan(fov_rad / 2)
    range_recip = 1 / (near - far)

    # 创建投影矩阵
    projection_matrix = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (near + far) * range_recip, 2 * near * far * range_recip],
        [0, 0, -1, 0],
    ])

    return projection_matrix

mesh = trimesh.load('cow.obj',force='mesh')
print(mesh.is_watertight)
scene = mesh.scene()

angle = np.pi / 4  # 旋转 45 度

# 创建一个旋转矩阵，围绕 z 轴旋转
rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])

# 将旋转矩阵应用到 Trimesh 对象上
mesh.apply_transform(rotation_matrix)


from trimesh_projection_copy import perspective_projected
path2d = perspective_projected(mesh,[0,0,-1],perspective_matrix(45,1,1,100))
fig, ax = plt.subplots()

# 绘制MultiPolygon
for polygon in path2d.geoms:
    x,y = polygon.exterior.xy
    ax.plot(x, y)

# 显示图形
plt.show()
print(111)