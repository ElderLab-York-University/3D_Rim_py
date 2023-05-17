import numpy as np
import pyrender
import trimesh
from pyrender.shader_program import ShaderProgramCache
from PIL import Image

renderer = pyrender.OffscreenRenderer(640, 480)
renderer._renderer._program_cache = ShaderProgramCache(shader_dir="shaders")
camera_pose = np.array(
    [[ 1,  0,  0,  0],
     [ 0,  0, -1, -10],
     [ 0,  1,  0,  0],
     [ 0,  0,  0,  1]]
)

scene = pyrender.Scene(bg_color=(0, 0, 0))
scene.add(pyrender.Mesh.from_trimesh(trimesh.primitives.Capsule(), smooth = False))
camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0, znear = 0.5, zfar = 40)
scene.add(camera, pose=camera_pose)
normals, depth = renderer.render(scene)
world_space_normals = normals / 255 * 2 - 1
print(normals[0][0])

image = Image.fromarray(normals, 'RGB')
image.show()