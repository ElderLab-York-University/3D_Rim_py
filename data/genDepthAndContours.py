import math
import os
import fnmatch
import threading
import imageio
from tqdm import tqdm
from data import renderObj
import numpy as np
import pyrender

lock = threading.Lock()
corrections = []
for i in range(0,12):
    corrections.append([])

def find_obj_files(path):
    obj_files = []

    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, '*.obj'):
            obj_files.append(os.path.join(root, filename))

    return obj_files

def convert(objPth,depthOutDir,dir_name):
    mesh = renderObj.loadAndScale(objPth)
    for fov in range(2,12):
        for i in range(0,10):
            depthOutFile = os.path.join(depthOutDir, "fov_pi_"+str(fov)+"_"+ str(i) + dir_name + ".npz")
            if not os.path.exists(depthOutFile):
                c1 = None
                try:
                    c1, c2, contours,rotations = renderObj.gen_contours_and_curvature(mesh,yfov=math.pi/fov)
                except:
                    pass
                if c1 is not None:
                    correction = np.corrcoef(c1,c2)[0][1]
                    corrections[fov].append(correction)
                    np.savez(depthOutFile, c1=contours, c2=c2,contours=contours,rotations=rotations,correction=correction)
            else:
                corrections[fov].append(np.load(depthOutFile)['correction'])

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
    objPth = "../ShapeNetCore_100"
    depthPth = "../ShapeNetCore_100_npz"
    print("start")
    obj_files = find_obj_files(objPth)
    print("finish reading dir")
    for obj_file in tqdm(obj_files):
        # 获取目录结构
        relative_path = os.path.relpath(obj_file, objPth)
        dir_name, _ = os.path.split(relative_path)
        os.makedirs(depthPth, exist_ok=True)
        dir_name = dir_name.replace('\\', '')
        convert(obj_file, depthPth,dir_name)
    for i,correction in enumerate(corrections):
        np.save('corrections'+str(i)+'.npy',correction)



