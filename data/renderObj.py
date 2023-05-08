import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt

def normalize_mesh(mesh, target_size=1.0):
    bbox = mesh.bounds  # 获取模型的边界框
    size = bbox[1] - bbox[0]  # 计算模型的尺寸
    max_size = size.max()  # 获取最大尺寸

    # 缩放模型以使最大尺寸等于目标尺寸
    scale_factor = target_size / max_size
    mesh.apply_scale(scale_factor)

    return mesh

def main():
    plyPath = './testData/cow.obj'
    loadedMesh = trimesh.load(plyPath)
    scaledMesh = normalize_mesh(loadedMesh)
    if isinstance(scaledMesh, trimesh.Trimesh):
        trimeshScene = trimesh.Scene()
        trimeshScene.add_geometry(scaledMesh)
    else:
        trimeshScene = scaledMesh
    scene = pyrender.Scene.from_trimesh_scene(trimeshScene)


    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
            [0.0, -s, s, 1],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, s, s, 1],
            [0.0, 0.0, 0.0, 1.0],
            ])

    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,innerConeAngle = np.pi / 16.0,outerConeAngle = np.pi / 6.0)
    scene.add(camera,pose=camera_pose)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=480, viewport_height=480)
    color, depth = r.render(scene)

    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()

if __name__ == "__main__":
    main()
