import os
import fnmatch
import threading
import imageio
from tqdm import tqdm
from data import renderObj
import numpy as np
import pyrender

lock = threading.Lock()


def find_obj_files(path):
    obj_files = []

    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, '*.obj'):
            obj_files.append(os.path.join(root, filename))

    return obj_files

def convert(objPth,depthOutDir):
    mesh = renderObj.loadAndScale(objPth)
    for i in range(0,360,60):
        depthOutFile = os.path.join(depthOutDir, "depth"+str(i)+".png")
        if os.path.exists(depthOutFile):
            continue
        else:
            depthImg = renderObj.render(r, mesh, r_angle=i)
            imageio.imwrite(depthOutFile, touint8(depthImg))
def touint8(array):
    try:
        normalized_array = (array - array.min()) / (array.max() - array.min())
        uint8_array = (normalized_array * 255).astype(np.uint8)
    except:
        pass
    return uint8_array

def processing(obj_file):
    depthPth = "../ShapeNetCore_Depth"
    objPth = "../ShapeNetCore.v2"
    relative_path = os.path.relpath(obj_file, objPth)
    dir_name, _ = os.path.split(relative_path)
    depthOutDir = os.path.join(depthPth, dir_name)
    os.makedirs(depthOutDir, exist_ok=True)
    try:
        convert(obj_file, depthOutDir)
    except:
        pass

if __name__ == "__main__":
    r = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=1024)
    objPth = "../ShapeNetCore.v2"
    depthPth = "../ShapeNetCore_Depth"
    print("start")
    obj_files = find_obj_files(objPth)
    print("finish reading dir")
    for obj_file in tqdm(obj_files):
        # 获取目录结构
        relative_path = os.path.relpath(obj_file, objPth)
        dir_name, _ = os.path.split(relative_path)
        depthOutDir = os.path.join(depthPth, dir_name)
        os.makedirs(depthOutDir, exist_ok=True)
        try:
            convert(obj_file, depthOutDir)
        except:
            pass


