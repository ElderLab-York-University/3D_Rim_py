from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import numpy as np


def rotate_and_center_vertices(vertices):
    # Compute the centroid
    hull = ConvexHull(vertices)
    centroid = np.mean([vertices[i] for i in hull.vertices], axis=0)

    # Move vertices to the origin
    vertices = vertices - centroid

    # Generate random rotation angles
    rotation_angles = np.random.rand(3) * 360

    # Create rotation matrix and apply to vertices
    rotation = R.from_euler('xyz', rotation_angles, degrees=True)
    vertices = rotation.apply(vertices)

    return vertices, rotation_angles


def project_and_compute_hull(vertices, fov):
    # Apply rotate_and_center_vertices function
    vertices, rotation_angles = rotate_and_center_vertices(vertices)

    # Perspective projection
    projected_vertices = []
    for vertex in vertices:
        x, y, z = vertex
        projected_vertex = [x / z * fov, y / z * fov]
        projected_vertices.append(projected_vertex)

    # Compute Convex Hull
    hull = ConvexHull(projected_vertices)

    # Create dictionary mapping hull vertices to their original index
    vertex_to_index = {tuple(vertex): index for index, vertex in enumerate(projected_vertices)}
    hull_indices = [vertex_to_index[tuple(projected_vertices[i])] for i in hull.vertices]

    return hull_indices, rotation_angles, vertex_to_index

