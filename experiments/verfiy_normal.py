import trimesh
import pyrender
import numpy as np

from data import renderObj

# 加载一个trimesh的mesh对象
mesh = trimesh.load('cow.obj',force='mesh')
c_mesh = mesh.copy()
c_mesh = renderObj.normalize_mesh(c_mesh, fov=np.pi/3)
c_mesh.remove_degenerate_faces()
# 创建一个pyrender的Mesh对象
r = pyrender.Mesh.from_trimesh(c_mesh)

# 创建一个Scene对象并将Mesh对象添加到场景中
scene = pyrender.Scene()
scene.add(r)

# 创建一个 PerspectiveCamera 对象
camera = pyrender.PerspectiveCamera(yfov=np.pi/3)

# 设置摄像机的位置
camera_pose = np.array([
   [1.0,  0.0, 0.0, 0.0],
   [0.0,  1.0, 0.0, 0.0],
   [0.0,  0.0, 1.0, 1.0],
   [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

# 创建一个OffscreenRenderer对象
renderer = pyrender.OffscreenRenderer(480, 480)

# 渲染场景
color, depth = renderer.render(scene)

# 显示渲染的图像
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,2,1)
plt.imshow(color)
plt.subplot(1,2,2)
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()