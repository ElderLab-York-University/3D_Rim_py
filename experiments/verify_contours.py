import matplotlib.pyplot as plt
import numpy as np
import pymeshfix
import trimesh
import matplotlib.cm as cm
from trimesh import Trimesh

from data import renderObj
from data.renderObj import rotate_trimesh
from experiments.cal_curvature import cal_curvature
from experiments.cal_curvature2 import cal_curvature2

mesh = trimesh.load('cow.obj',force='mesh')

c_mesh = mesh.copy()
c_mesh = renderObj.normalize_mesh(c_mesh, fov=np.pi/3)
rotate_angle = np.asarray([ 90,0,0])
rotatedMesh = rotate_trimesh(c_mesh,[0,1,0],rotate_angle[0])
rotatedMesh = rotate_trimesh(rotatedMesh,[1,0,0],rotate_angle[1])
rotatedMesh = rotate_trimesh(rotatedMesh,[0,0,1],rotate_angle[2])
c_mesh = rotatedMesh
c_mesh.remove_degenerate_faces()

'''
v = c_mesh.vertices
f = c_mesh.faces

meshfix = pymeshfix.MeshFix(v, f)
fixed_mesh = Trimesh(vertices=v,faces=f)
bool = fixed_mesh.fill_holes()
print(bool)
'''
c1,c2,contours,valid_map,angle,v3d = cal_curvature2(c_mesh,np.pi/3)

c1 = np.asarray(c1)
c2 = np.asarray(c2)
c1_1 = c1[valid_map]
c2_2 = c2[valid_map]
fig, ax = plt.subplots(dpi=1000)

x, y = contours[:,0],contours[:,1]
for i in range(len(x)):
    x1 = x[i]
    y1 = y[i]
    x0 = x[i-1]
    y0 = y[i-1]
    #plt.quiver(x0,y0,x1-x0,y1-y0)

# 创建一个散点图，其中点的颜色是根据z值来的
vmax = np.max(np.abs(c1))
plt.plot(x,y,linewidth =0.6,color='black')
ax.axis('equal')
# 添加一个颜色条
plt.title('projected rim')

plt.show()



