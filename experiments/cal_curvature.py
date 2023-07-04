import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from trimesh.intersections import mesh_plane
from sklearn.decomposition import PCA
import numpy as np
from shapely import Polygon, union_all


def calculate_angle(x1, y1, x2, y2):
    vector1 = np.array([x1, y1])
    vector2 = np.array([x2, y2])

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = np.arccos(cos_angle)

    # 将弧度转化为角度
    return np.degrees(angle)
def compute_camera_matrices(fov, camera_position):
    # 首先，定义图像平面的大小
    img_width = 640
    img_height = 480

    # 计算焦距
    f = img_width / (2 * np.tan(fov / 2))

    # 内参矩阵
    K = np.array([[f, 0, img_width / 2],
                  [0, f, img_height / 2],
                  [0, 0, 1]])

    # 外参矩阵
    # 我们假设相机看向 -Z 方向，并且相机的 "上" 方向是 +Y 方向。
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]])

    T = -R @ camera_position

    RT = np.hstack([R, T.reshape(3, 1)])

    return K, RT
def perspective_matrix(fov, aspect, near, far):
    # fov 应该是以度为单位，将其转换为弧度
    fov_rad = np.deg2rad(fov)

    # 计算透视投影矩阵的元素
    f = 1 / np.tan(fov_rad / 2)
    range_recip = 1 / (near - far)

    # 创建投影矩阵
    projection_matrix = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (near + far) * range_recip, 2 * near * far * range_recip],
        [0, 0, -1, 0],
    ])

    return projection_matrix
def projected(mesh,
              normal,
              matrix,
              **kwargs):
    from trimesh.path import Path2D
    from trimesh.exchange.load import load_path

    projection = projected(
        mesh=mesh, normal=normal,matrix=matrix,**kwargs)
    if projection is None:
        return Path2D()
    return load_path(projection)
def perspective_projected(mesh,
              K,RT,):

    vertices_2D = map_points_to_image(mesh.vertices,K,RT)
    polygons = []
    for face in mesh.faces:
        vlist = [vertices_2D[face[0]],vertices_2D[face[1]],vertices_2D[face[2]]]
        trig = Polygon(vlist)
        polygons.append(trig)
    final = union_all(polygons)

    return final,mesh.edges_sorted,vertices_2D
def map_points_to_image(points_3d, K, RT):
    # 把点从世界坐标系变换到相机坐标系
    points_cam = points_3d @ RT[:, :3].T + RT[:, 3]

    # 把点从相机坐标系映射到图像平面
    points_2d = points_cam @ K.T

    # 进行透视除法，使得点的坐标是归一化的齐次坐标
    div = points_2d[:, 2, np.newaxis]
    points_2d = points_2d / div

    return points_2d[:, :2]  # 只返回x和y坐标

def cal_curvatrue(rotatedMesh,fov):
    mesh = rotatedMesh
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    camera_pose = np.array([0, 0, 1]) #look at 0,0,0
    K, RT = compute_camera_matrices(fov, camera_pose)
    path2d, edges, vertices_2D = perspective_projected(mesh, K, RT)
    # contours = edge_to_contours(vertices_2D,edges)
    if hasattr(path2d,'exterior'):
        x_, y_ = path2d.exterior.xy
        A = np.array(vertices_2D)
        B = np.array(path2d.exterior.xy)
        B = B.T
        A1 = A[:, 0]
        B1 = B[:, 0]
        sort_idx_1 = A1.argsort()
        x_index = sort_idx_1[np.searchsorted(A1, B1, sorter=sort_idx_1)]
        x_final = np.zeros_like(x_index)
        mesh_vertex_faces = mesh.vertex_faces
        mesh_vertex_3d = mesh.vertices
        mesh_faces = mesh.faces
        for i, x in enumerate(x_index):
            point = A[x]
            contour_point = B[i]
            if np.allclose(point, contour_point):
                x_final[i] = x
            else:
                x_final[i] = -1

        z_axis = np.array([0, 0, -1])
        curvature_tangent_surface = []
        curvature_contours_surface = []
        for idx, i in enumerate(x_final):
            p1 = idx - 1
            p2 = idx
            p3 = (idx + 1) % len(x_final)
            x1 = x_[p1] - x_[p2]
            x2 = x_[p3] - x_[p2]
            y1 = y_[p1] - y_[p2]
            y2 = y_[p3] - y_[p2]

            angle = calculate_angle(x1, x2, y1, y2)
            arclen = np.linalg.norm([x1, y1]) + np.linalg.norm([x2, y2])
            curvature_contours_surface.append(angle / arclen)
            if i == -1:
                curvature_tangent_surface.append(-1)
            else:
                vertex = mesh_vertex_3d[i]
                faces = mesh_vertex_faces[i]
                lines = []
                v = vertex - camera_pose
                plane_normal = z_axis - (np.dot(z_axis, v) / np.dot(v, v)) * v
                for f_index in faces:
                    if f_index == -1:
                        pass
                    else:
                        face = mesh_faces[f_index]
                        def_face = [[0, 1, 2]]
                        tri_vertices = [mesh_vertex_3d[face[0]], mesh_vertex_3d[face[1]], mesh_vertex_3d[face[2]]]
                        triangle = trimesh.Trimesh(vertices=tri_vertices, faces=def_face)
                        w = np.cross(plane_normal, v)
                        line = mesh_plane(mesh=triangle, plane_normal=w, plane_origin=vertex)
                        if line.shape[0] > 0:
                            lines.append(line[0])
                vectors = []
                for line in lines:
                    v = line - vertex;
                    if (v[0]==0).all():
                        vectors.append(v[1])
                    else:
                        vectors.append(v[0])

                if len(vectors) > 0:
                    max_neg = 0
                    max_pos = 0
                    max_ang_neg = 0
                    max_ang_pos = 0
                    v0 = vectors[0]
                    for i in range(1, len(vectors)):
                        vi = vectors[i]
                        cross = np.dot(np.cross(v0, vi), w)
                        dot = abs(np.dot(v0, vi))
                        if cross > 0 and dot > max_ang_pos:
                            max_pos = i
                            max_ang_pos = dot
                        elif cross < 0 and dot > max_ang_neg:
                            max_neg = i
                            max_ang_neg = dot
                        else:
                            pass
                    angle = max_ang_pos + max_ang_neg
                    arclen = np.linalg.norm(vectors[max_pos]) + np.linalg.norm(vectors[max_neg])

                    curvature_tangent_surface.append(angle / arclen)
                else:
                    curvature_tangent_surface.append(-1)
        curvature_tangent_surface_final = []
        curvature_contours_surface_final = []
        for i in range(0,len(curvature_tangent_surface)):
            if curvature_tangent_surface[i] == -1 or curvature_contours_surface[i] == -1:
                pass
            else:
                curvature_tangent_surface_final.append(curvature_tangent_surface[i])
                curvature_contours_surface_final.append(curvature_contours_surface[i])

        return curvature_contours_surface_final,curvature_tangent_surface_final,path2d.exterior.xy
    else:
        return None,None,None
