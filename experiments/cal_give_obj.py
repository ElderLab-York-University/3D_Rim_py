import os

import numpy as np
import trimesh

from data import renderObj

shape_net_core_path = '../ShapeNetCore.v2'
sample_path = '../ShapeNetCoreSample2'
max_num_sample_pre_class = 10
max_num_sample_pre_obj = 10
num_class = 55

def cal(objPth):
    fov_list = [90,30,15, 8, 4, 2]
    mesh = trimesh.load(objPth,force='mesh')
    for fov in fov_list:
        c_mesh = mesh.copy()
        c_mesh = renderObj.normalize_mesh(c_mesh,np.pi/fov)
        c_mesh.remove_degenerate_faces()
        for i in range(max_num_sample_pre_obj):
            abspath = os.path.dirname(objPth)
            relative_path = os.path.relpath(abspath, shape_net_core_path)
            sample_save_path = os.path.join(sample_path, relative_path)
            filename = str(fov) + '_' + str(i) + 'npz'
            os.makedirs(sample_save_path, exist_ok=True)
            sample_save_path = os.path.join(sample_save_path, filename)
            if not os.path.exists(sample_save_path):
                c1 = None
                try:
                    c1, c2, contours, rotations = renderObj.gen_contours_and_curvature(c_mesh, yfov=np.pi/fov)
                except:
                    pass
                if c1 is not None:
                    np.savez(sample_save_path, c1=c1, c2=c2, contours=contours, rotations=rotations)
            else:
                pass