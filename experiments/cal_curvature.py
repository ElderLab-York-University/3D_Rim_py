import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely
import trimesh
import matplotlib.pyplot as plt
from shapely.geometry import polygon
from trimesh.intersections import mesh_plane
from sklearn.decomposition import PCA
import numpy as np
from shapely import Polygon, union_all
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
from skspatial.objects import Vector
def normalize(v):
    # 计算向量的长度（范数）
    norm = np.linalg.norm(v)
    # 避免除以0
    if norm == 0:
       return v
    # 返回归一化的向量
    return v / norm

def compute_camera_matrices(fov, camera_position):
    # 首先，定义图像平面的大小
    img_width = 480
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
def calculate_angle(vec1, vec2):
    # 计算两个向量的点积
    dot_product = np.dot(vec1, vec2)

    # 计算两个向量的模
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    # 计算两个向量的夹角
    cos_theta = dot_product / (vec1_norm * vec2_norm)
    angle_in_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle_in_radians
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
    '''
    fig, ax = plt.subplots(dpi=1000)
    debug
    '''
    for face in mesh.faces:
        vlist = [vertices_2D[face[0]],vertices_2D[face[1]],vertices_2D[face[2]]]
        trig = Polygon(vlist)
        polygons.append(trig)
        '''
        draw_trig = [vertices_2D[face[0]],vertices_2D[face[1]],vertices_2D[face[2]],vertices_2D[face[0]]]
        draw_trig = np.asarray(draw_trig)
        ax.plot(draw_trig[:,0],draw_trig[:,1],linewidth=0.1)
        '''
    '''
    ax.scatter(vertices_2D[:,0],vertices_2D[:,1],s=0.1,edgecolors='none')
    ax.axis('equal')
    plt.title('projected triangles')
    plt.show()
    '''
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
def project_point_to_plane(point, plane_point, plane_normal):
    # 标准化法向量
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # 计算点到平面的距离
    d = np.dot(np.subtract(point, plane_point), plane_normal)
    # 计算投影点
    projected_point = np.subtract(point, np.multiply(d, plane_normal))
    return projected_point

def cal_curvature(rotatedMesh, fov):
    mesh = rotatedMesh
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    camera_pose = np.array([0, 0, 1]) #look at 0,0,0
    K, RT = compute_camera_matrices(fov, camera_pose)
    valid_polygon = True

    try:
        path2d, edges, vertices_2D = perspective_projected(mesh, K, RT)
    except:
        valid_polygon = False
    if valid_polygon:
        # contours = edge_to_contours(vertices_2D,edges)
        if isinstance(path2d, shapely.Polygon):
            path2d = polygon.orient(path2d, sign=1.0)
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
            mesh_vertex_normals = mesh.vertex_normals
            coutour_filter = []
            for i, x in enumerate(x_index):
                point = A[x]
                contour_point = B[i]
                if np.allclose(point, contour_point):
                    x_final[i] = x
                    coutour_filter.append(True)
                else:
                    x_final[i] = -1
                    coutour_filter.append(False)
            coutour_filter = np.asarray(coutour_filter)
            B = B[coutour_filter]
            x_final = x_final[coutour_filter]
            z_axis = np.array([0, 0, -1])
            curvature_apparent = []
            curvature_normal = []
            apparent_angle = []
            valid_map = []
            vertex_3d = mesh_vertex_3d[x_final]
            for idx, i in enumerate(x_final):
                vertex = mesh_vertex_3d[i]
                view_vector = vertex - camera_pose
                vertex_normal = mesh_vertex_normals[i]
                N_plane_normal = np.cross(view_vector, vertex_normal)
                if idx == len(x_final)-1:
                    break
                '''
                plane P:
                defined by point P and normal at view vector 
                plane N:
                defined by point P and normal at N_plane_normal
                '''

                p1_idx = idx - 1

                if p1_idx == -1:
                    p1_idx = -2

                p2_idx = idx

                p3_idx = (idx + 1) % len(x_final)


                p1 = B[p1_idx]
                p2 = B[p2_idx]
                p3 = B[p3_idx]

                V1 = p2 - p1
                V2 = p3 - p2
                V1 = [V1[0], V1[1], 0]
                V2 = [V2[0], V2[1], 0]
                V1 = np.asarray(V1)
                V2 = np.asarray(V2)
                V1 = V1/480
                V2 = V2/480
                arclen = np.linalg.norm(V1) + np.linalg.norm(V2)

                V1 = normalize(V1)
                V2 = normalize(V2)

                normal = [0, 0, 1]
                a1 = np.cross(V1, V2).dot(normal)
                a2 = np.dot(V1, V2)
                angle = np.arctan2(a1, a2)
                apparent_angle.append(angle)
                apparent_value = angle / (arclen / 2)

                curvature_apparent.append(apparent_value)






                '''valid_map
                RMS distance of the point 
                '''

                '''
                find intersection of groundtruth 3d mesh with plane N
                '''
                if i == -1:
                    raise RuntimeError("wrong index")
                else:

                    vertex = mesh_vertex_3d[i]
                    faces = mesh_vertex_faces[i]
                    lines = []
                    '''
                    for each triangle 
                    '''
                    for f_index in faces:
                        if f_index == -1:
                            pass
                        else:
                            face = mesh_faces[f_index]
                            def_face = [[0, 1, 2]]
                            tri_vertices = [mesh_vertex_3d[face[0]], mesh_vertex_3d[face[1]], mesh_vertex_3d[face[2]]]
                            triangle = trimesh.Trimesh(vertices=tri_vertices, faces=def_face)
                            line = mesh_plane(mesh=triangle, plane_normal=N_plane_normal, plane_origin=vertex)
                            if line.shape[0] > 0:
                                lines.append(line[0])
                    vectors = []
                    for line in lines:
                        v = line - vertex;
                        if (v[0] == 0).all():
                            vectors.append(v[1])
                        else:
                            vectors.append(v[0])

                    if len(vectors) == 2:
                        '''
                        max_neg = 0
                        max_pos = 0
                        max_ang_neg = 0
                        max_ang_pos = 0
                        v0 = vectors[0]
                        for i in range(1, len(vectors)):
                            vi = vectors[i]
                            cross = np.dot(np.cross(v0, vi), N_plane_normal)
                            dot = abs(np.dot(v0, vi))
                            if cross > 0 and dot > max_ang_pos:
                                max_pos = i
                                max_ang_pos = dot
                            elif cross < 0 and dot > max_ang_neg:
                                max_neg = i
                                max_ang_neg = dot
                            else:
                                pass
                        '''
                        angle = np.pi - calculate_angle(vectors[0], vectors[1])
                        arclen = np.linalg.norm(vectors[0]) + np.linalg.norm(vectors[1])
                        normal_value = angle / (arclen / 2)
                        curvature_normal.append(normal_value)
                        valid_map.append(True)
                    else:
                        curvature_normal.append(0)
                        valid_map.append(False)
                        debug = False
                        if debug:
                            fig = plt.figure(dpi=600)
                            ax = fig.add_subplot(111, projection='3d')

                            max_x, min_x = -1, 1
                            max_y, min_y = -1, 1
                            max_z, min_z = -1, 1

                            trilist = []
                            for f_index in faces:
                                if f_index == -1:
                                    pass
                                else:
                                    face = mesh_faces[f_index]
                                    tri_vertices = [np.asarray(mesh_vertex_3d[face[0]]),
                                                    np.asarray(mesh_vertex_3d[face[1]]),
                                                    np.asarray(mesh_vertex_3d[face[2]])]
                                    tri_vertices = np.asarray((tri_vertices))
                                    max_x = max(tri_vertices[:, 0].max(), max_x)
                                    min_x = min(tri_vertices[:, 0].min(), min_x)

                                    max_y = max(tri_vertices[:, 1].max(), max_y)
                                    min_y = min(tri_vertices[:, 1].min(), min_y)

                                    max_z = max(tri_vertices[:, 2].max(), max_z)
                                    min_z = min(tri_vertices[:, 2].min(), min_z)

                                    tri = Poly3DCollection([tri_vertices], alpha=0.5)
                                    tri.set_color(colors.rgb2hex(np.random.rand(3)))
                                    trilist.append(tri_vertices.flatten())
                                    ax.add_collection3d(tri)

                            normal = N_plane_normal
                            v_normal = vertex_normal
                            '''
                            d = -point.dot(normal)
    
                            nx, ny = (50, 50)
                            x = np.linspace(max(max_x,max_y,max_z)*2-(max(max_x,max_y,max_z)-min(min_x,min_y,min_z))*4, max(max_x,max_y,max_z)*2, nx)
                            y = np.linspace(max(max_x,max_y,max_z)*2-(max(max_x,max_y,max_z)-min(min_x,min_y,min_z))*4, max(max_x,max_y,max_z)*2, ny)
    
                            xx, yy = np.meshgrid(x, y)
    
                            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    
                            ax.plot_surface(xx, yy, z,alpha=0.2)
                            '''
                            ax.quiver(vertex[0], vertex[1], vertex[2], normal[0], normal[1], normal[2], length=0.01,
                                      normalize=True, color='blue')
                            ax.quiver(vertex[0], vertex[1], vertex[2], v_normal[0], v_normal[1], v_normal[2], length=0.01,
                                      normalize=True, color='red')
                            ax.quiver(vertex[0], vertex[1], vertex[2], view_vector[0], view_vector[1], view_vector[2],
                                      length=0.01, normalize=True, color='green')

                            ax.scatter(vertex[0], vertex[1], vertex[2], marker='*', color='red', s=2)

                            ax.set_aspect('equal')

                            ax.set_xlim([min_x, max_x])
                            ax.set_ylim([min_y, max_y])
                            ax.set_zlim([min_z, max_z])

                            ax.plot([], [], [], color="r", label="vertex normal")
                            ax.plot([], [], [], color="g", label="view_vector")
                            ax.plot([], [], [], color="b", label="plane normal")
                            ax.set_xlabel("x")
                            ax.set_ylabel("y")
                            ax.set_zlabel("z")
                            ax.legend()
                            plt.title("virtualization of normal curvature computation")

                            plt.show()

            '''
            #fiter out invaild result
            for i in range(0,len(curvature_apparent)):
                if curvature_apparent[i] == -1 or curvature_normal[i] == -1:
                    pass
                else:
                    curvature_tangent_surface_final.append(curvature_apparent[i])
                    curvature_contours_surface_final.append(curvature_normal[i])
            '''
            return curvature_apparent, curvature_normal, B, valid_map,apparent_angle,vertex_3d
        else:
            return None, None, None, None, None,None
    else:
        return None, None, None, None, None,None