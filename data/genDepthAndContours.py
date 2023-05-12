import os
import fnmatch

import imageio
from tqdm import tqdm

from data import renderObj
import numpy as np


def find_obj_files(path):
    obj_files = []

    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, '*.obj'):
            obj_files.append(os.path.join(root, filename))

    return obj_files

def convert(objPth,depthOutDir):
    for i in range(0,360,60):
        depthOutFile = os.path.join(depthOutDir, "depth"+str(i)+".png")
        if os.path.exists(depthOutFile):
            continue
        else:
            depthImg = renderObj.render(objPth,r_angle=i)
            imageio.imwrite(depthOutFile, touint8(depthImg))


def touint8(array):
    normalized_array = (array - array.min()) / (array.max() - array.min())
    uint8_array = (normalized_array * 255).astype(np.uint8)


    return uint8_array

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
    except Exception as e:
        continue
