import trimesh
import pyrender
import numpy as np

from data import renderObj
from cal_curvature import  compute_camera_matrices,perspective_projected
from data.renderObj import rotate_trimesh

# 加载一个trimesh的mesh对象
mesh = trimesh.load('cow.obj',force='mesh')
c_mesh = mesh.copy()
c_mesh = renderObj.normalize_mesh(c_mesh, fov=np.pi/3)
rotate_angle = np.asarray([ 90,0,0])
rotatedMesh = rotate_trimesh(c_mesh,[0,1,0],rotate_angle[0])
rotatedMesh = rotate_trimesh(rotatedMesh,[1,0,0],rotate_angle[1])
rotatedMesh = rotate_trimesh(rotatedMesh,[0,0,1],rotate_angle[2])
c_mesh = rotatedMesh
c_mesh.remove_degenerate_faces()

# 创建一个pyrender的Mesh对象
r = pyrender.Mesh.from_trimesh(c_mesh)

# 创建一个Scene对象并将Mesh对象添加到场景中
scene = pyrender.Scene()
scene.add(r)
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)

# 创建一个 PerspectiveCamera 对象
fov = np.pi/3
camera = pyrender.PerspectiveCamera(yfov=fov)

# 设置摄像机的位置
camera_pose = np.array([
   [1.0,  0.0, 0.0, 0.0],
   [0.0,  1.0, 0.0, 0.0],
   [0.0,  0.0, 1.0, 1.0],
   [0.0,  0.0, 0.0, 1.0],
])
scene.add(light, pose=camera_pose)
camera_pose_3d = np.array([0, 0, 1]) #look at 0,0,0
K, RT = compute_camera_matrices(np.pi/3, camera_pose_3d)
path2d, edges, vertices_2D = perspective_projected(c_mesh, K, RT)

scene.add(camera, pose=camera_pose)
# 创建一个OffscreenRenderer对象
renderer = pyrender.OffscreenRenderer(1920, 1920)
# 渲染场景
color, depth = renderer.render(scene)
x,y = path2d.exterior.xy
# 显示渲染的图像
import matplotlib.pyplot as plt

plt.figure(dpi=1200)
plt.subplot(1,2,1)
plt.imshow(color)
plt.axis('off')
plt.axis('equal')
plt.subplot(1,2,2)
plt.plot(x,y)
plt.axis('off')
plt.axis('equal')
plt.show()