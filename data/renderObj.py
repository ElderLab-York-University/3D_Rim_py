import math
import alphashape
import pyrender
import scipy.spatial
import trimesh
import numpy as np


def normalize_mesh(mesh,fov):
    # rewrite from yiming's code
    X = mesh.vertices[:,0]
    Y = mesh.vertices[:,1]
    Z = mesh.vertices[:,2]

    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    R = np.sqrt(X**2+Y**2+Z**2)
    r = R.max()
    excpet_r = math.sin(fov/2)
    mesh.apply_scale(excpet_r/r)
    return mesh
def fov_err(a,X,Y,Z,fov):
    Z = a*(Z-1)+1
    X = a*X
    Y = a*Y
    x = np.divide(X,Z)
    y = np.divide(Y,Z)

    points = np.asarray((x,y)).T
    alpha_shape = alphashape.alphashape(points,0.8)
    centroid = alpha_shape.centroid

    x = x-centroid.x
    y = y-centroid.y
    R = np.sqrt(x**2+y**2)
    if Z.min() <=0:
        return math.inf
    else:
        return (np.arctan(R).max()-fov/2)**2






def compute_mesh_center(mesh):
    vex3d = np.asarray(mesh.vertices)
    vex2d = vex3d[:,0:2]
    K = scipy.spatial.ConvexHull(vex2d)
    center = np.zeros(3)
    points = vex3d[K.vertices]
    center[0] = np.mean(points[:,0])
    center[1] = np.mean(points[:,1])
    center[2] = np.mean(vex3d[:,2])
    return center

def rotate_trimesh(mesh, axis, angle):

    # Convert the angle to radians
    angle_rad = np.radians(angle)

    # Create the rotation matrix
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, axis)

    # Apply the rotation to the mesh
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)

    return rotated_mesh
def create_point_cloud_mesh(points, colors=None, point_size=0.005):
    if colors is None:
        colors = np.full(points.shape, [0.5, 0.5, 0.5], dtype=np.float32)

    point_cloud = pyrender.Node(
        mesh=pyrender.Mesh.from_points(points, colors=colors),
        scale=np.array([point_size, point_size, point_size]),
    )
    return point_cloud
def normalize(vector):
    return vector / np.linalg.norm(vector)

def camera_matrix(camera_position, target_position, up_direction):
    forward = normalize(camera_position - target_position)
    temp_right = np.cross(up_direction, forward)
    right = normalize(temp_right)
    up = np.cross(forward, right)

    rotation_matrix = np.array([right, up, forward])
    translation_vector = -np.dot(rotation_matrix, camera_position)

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = rotation_matrix.T
    view_matrix[:3, 3] = translation_vector

    return view_matrix










