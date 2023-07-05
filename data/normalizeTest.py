import trimesh

from data import renderObj

mesh = trimesh.load('../experiments/cow.obj', force='mesh')
mesh = renderObj.normalize_mesh(mesh,60)
print(1)