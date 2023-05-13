import math
import imageio
import numpy
import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import cv2



def normalize_mesh(mesh):
    # Translate the mesh to the origin
    mesh_center = mesh.centroid
    mesh.apply_translation(-mesh_center)

    # Scale the mesh to have a uniform size (e.g., the longest dimension equals 1)
    mesh_bounds = mesh.bounds
    mesh_size = mesh_bounds[1] - mesh_bounds[0]
    max_dimension = mesh_size.max()
    mesh.apply_scale(1 / max_dimension)


    return mesh
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
def find_occluding_contours(color, depth, depth_threshold=0.01, canny_low_threshold=100, canny_high_threshold=200):
    # Normalize depth image
    depth_normalized = np.copy(depth)
    depth_normalized[depth_normalized!=0] = 1
    depth_normalized = (depth_normalized * 255).astype(np.uint8)
    # Apply Canny edge detection on the depth image
    edges = cv2.Canny(depth_normalized, canny_low_threshold, canny_high_threshold)

    # Dilate the edges to make them more prominent
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Create a mask based on the depth discontinuity
    depth_diff = cv2.absdiff(depth, cv2.GaussianBlur(depth, (5, 5), 0))
    depth_mask = (depth_diff > depth_threshold).astype(np.uint8) * 255

    # Combine the edges and the depth mask to obtain the occluding contours
    occluding_contours = cv2.bitwise_and(edges, depth_mask)

    # Optionally, overlay the occluding contours on the original color image
    color_with_contours = color.copy()
    color_with_contours[occluding_contours > 0] = [0, 255, 0]

    return occluding_contours, color_with_contours
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


def render(r,objpth,yfov=np.pi/3.0,aspectRatio=1.0,camera_pose=None,r_axis=[0,1,0],r_angle=0):
    loadedMesh = trimesh.load(objpth)
    scaledMesh = normalize_mesh(loadedMesh)
    rotatedMesh = rotate_trimesh(scaledMesh,r_axis,r_angle)
    if isinstance(rotatedMesh,trimesh.points.PointCloud):
        points = rotatedMesh.vertices
        colors = rotatedMesh.colors
        point_cloud_mesh = create_point_cloud_mesh(points, colors)
        scene = pyrender.Scene()
        scene.add_node(point_cloud_mesh)
    else:
        if isinstance(rotatedMesh, trimesh.Trimesh):
            trimeshScene = trimesh.Scene()
            trimeshScene.add_geometry(rotatedMesh)
        else:
            trimeshScene = rotatedMesh
        scene = pyrender.Scene.from_trimesh_scene(trimeshScene)


    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspectRatio)


    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,innerConeAngle = np.pi / 16.0,outerConeAngle = np.pi / 6.0)
    if camera_pose is not None:
        scene.add(camera, pose=camera_pose)
    else:
        camera_distance = 0.6 / math.tan(yfov / 2)+0.5
        default_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, camera_distance],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=default_pose)

    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)
   # occluding_contours, color_with_contours = find_occluding_contours(color, depth)

    return depth

if __name__ == "__main__":
    plyPath = './testData/cow.obj'
    depth = render(plyPath,r_angle=90)
    imageio.imwrite("./testData/cow2.png",depth)


