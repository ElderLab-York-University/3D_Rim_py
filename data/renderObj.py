import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import cv2


default_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 2.0],
    [0.0, 0.0, 0.0, 1.0],
])
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
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
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

def generate_camera_pose(distance, azimuth, elevation):
    # Convert angles from degrees to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    # Calculate the camera position in Cartesian coordinates
    x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = distance * np.sin(elevation_rad)
    z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

    camera_position = np.array([x, y, z])

    # Calculate the camera orientation using the look-at algorithm
    target = np.array([0, 0, 0])  # Camera is always looking at (0, 0, 0)
    up = np.array([0, 1, 0]) if np.abs(elevation) != 90 else np.array([0, 0, 1])

    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.vstack((left, up, forward)).T
    camera_pose[:3, 3] = camera_position
    print(camera_pose)
    return camera_pose


def render(objpth,yfov=np.pi/3.0,aspectRatio=1.0,camera_pose=default_pose):
    loadedMesh = trimesh.load(objpth)
    scaledMesh = normalize_mesh(loadedMesh)
    if isinstance(scaledMesh,trimesh.points.PointCloud):
        points = scaledMesh.vertices
        colors = scaledMesh.colors
        point_cloud_mesh = create_point_cloud_mesh(points, colors)
        scene = pyrender.Scene()
        scene.add_node(point_cloud_mesh)
    else:
        if isinstance(scaledMesh, trimesh.Trimesh):
            trimeshScene = trimesh.Scene()
            trimeshScene.add_geometry(scaledMesh)
        else:
            trimeshScene = scaledMesh
        scene = pyrender.Scene.from_trimesh_scene(trimeshScene)


    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspectRatio)


    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,innerConeAngle = np.pi / 16.0,outerConeAngle = np.pi / 6.0)
    scene.add(camera,pose=camera_pose)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=1080, viewport_height=1080)
    color, depth = r.render(scene)

    occluding_contours, color_with_contours = find_occluding_contours(color, depth)

    plt.figure(figsize=(24, 8))
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(occluding_contours,cmap=plt.cm.gray_r)
    plt.show()

if __name__ == "__main__":
    plyPath = './testData/trumpet.obj'
    render(plyPath)
