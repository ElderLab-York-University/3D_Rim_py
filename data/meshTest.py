import trimesh
from renderObj import compute_mesh_center
plyPath = './testData/cow.obj'

mesh = trimesh.load(plyPath,force='mesh')
print(type(mesh))
print(mesh.vertices)
print(mesh.bounds)