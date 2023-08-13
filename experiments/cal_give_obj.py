import os

import numpy as np
import pywavefront
import trimesh
import logging

from data import renderObj

shape_net_core_path = '../ShapeNetCore.v2'
sample_path = '../ShapeNetCoreSampleFinal'
max_num_sample_pre_class = 10
max_num_sample_pre_obj = 10
num_class = 55


def cal(objPth):
    fov_list = [4]
    try:
        wavefront_obj = pywavefront.Wavefront(objPth, collect_faces=True)
        mesh = trimesh.Trimesh(vertices=wavefront_obj.vertices, faces=wavefront_obj.mesh_list[0].faces)
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
                    c1, c2, contours, rotations,vaild_map,angle,vertex_3d = renderObj.gen_contours_and_curvature(c_mesh, yfov=np.pi/fov)
                    if c1 is not None:
                        np.savez(sample_save_path, c1=c1, c2=c2, contours=contours, rotations=rotations,vaild_map=vaild_map,angle=angle,vertex_3d=vertex_3d)
                else:
                    pass
    except:
        pass