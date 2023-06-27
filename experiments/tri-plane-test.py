import trimesh
from trimesh.intersections import *
vertices = [[-1,-1,-1],[1,-1,-1],[0,1,1]]
faces = [[0,1,2]]

mesh = trimesh.Trimesh(vertices=vertices,faces=faces)

lines = mesh_plane(mesh=mesh,plane_normal=[0,1,0],plane_origin=[0,0,0])
print(lines)
