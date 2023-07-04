import trimesh

from data import renderObj

mesh = trimesh.load('cow.obj',force='mesh')
mesh = renderObj.normalize_mesh(mesh,60)
print(1)