import pywavefront
import numpy as np
from scipy.spatial import ConvexHull
import networkx as nx

def readVandface(path):
    scene = pywavefront.Wavefront(path, collect_faces=True)
    vertex_to_faces = {}

    # 遍历每个 mesh
    for name, mesh in scene.meshes.items():
        # 遍历每个面
        for face in mesh.faces:
            # 遍历这个面的每个顶点
            for vertex_index in face:
                # 如果这个顶点还没有在字典中，就添加一个新的空列表
                if vertex_index not in vertex_to_faces:
                    vertex_to_faces[vertex_index] = []
                # 将这个面添加到这个顶点的相邻面列表中
                vertex_to_faces[vertex_index].append(face)

    vertex_index = 0
    adjacent_faces = vertex_to_faces[vertex_index]
    return scene,adjacent_faces



scene,adjacent_faces = readVandface("testData/cow.obj")
vertex = np.array(scene.vertices)
K = ConvexHull(vertex[:,0:2])
mean = np.mean(vertex[K.vertices],axis=0)
mean[2] = np.mean(vertex,axis=0)[2]



