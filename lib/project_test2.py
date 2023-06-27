import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from trimesh.intersections import mesh_plane
from sklearn.decomposition import PCA
from trimesh_projection_copy import perspective_projected

from edge_to_contours import edge_to_contours
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

'''
def are_points_in_same_plane(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    eigenvalues = np.sort(pca.explained_variance_)
    return np.isclose(eigenvalues[0], 0, atol=1e-8)
'''

mesh = trimesh.load('cow.obj',force='mesh')
mesh.merge_vertices()
mesh.remove_duplicate_faces()
#mesh = trimesh.primitives.Sphere()
scene = mesh.scene()

angle = np.pi / 4  # 旋转 45 度

# 创建一个旋转矩阵，围绕 z 轴旋转
rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])

# 将旋转矩阵应用到 Trimesh 对象上
mesh.apply_transform(rotation_matrix)

camera_pose = np.array([0,0,10])
K,RT = compute_camera_matrices(math.pi/4,camera_pose)
path2d,edges,vertices_2D = perspective_projected(mesh,K,RT)
#contours = edge_to_contours(vertices_2D,edges)
x_,y_ = path2d.exterior.xy
A = np.array(vertices_2D)
B = np.array(path2d.exterior.xy)
B = B.T
A1 = A[:,0]
B1 = B[:,0]
A2 = A[:,1]
B2 = B[:,1]
sort_idx_1 = A1.argsort()
x_index = sort_idx_1[np.searchsorted(A1,B1,sorter = sort_idx_1)]
x_final = np.zeros_like(x_index)
mesh_vertex_faces = mesh.vertex_faces
mesh_vertex_3d = mesh.vertices
mesh_faces = mesh.faces
for i,x in enumerate(x_index):
    point = A[x]
    contour_point = B[i]
    if np.allclose(point,contour_point) :
        x_final[i] = x
    else:
        x_final[i] = -1

z_axis = np.array([0,0,-1])
curvature_tangent_surface = []
curvature_contours_surface = []
for idx,i in enumerate(x_final):
    p1 = idx-1
    p2 = idx
    p3 = (idx+1)%len(x_final)
    x1 = x_[p1]-x_[p2]
    x2 = x_[p3]-x_[p2]
    y1 = y_[p1]-y_[p2]
    y2 = y_[p3]-y_[p2]

    angle = calculate_angle(x1,x2,y1,y2)
    arclen = np.linalg.norm([x1,y1])+np.linalg.norm([x2,y2])
    if angle == 0:
        print(idx)
    curvature_contours_surface.append(angle/arclen)
    if i == -1:
        curvature_tangent_surface.append(-1)
    else:
        vertex = mesh_vertex_3d[i]
        faces = mesh_vertex_faces[i]
        lines = []
        v = vertex-camera_pose
        plane_normal = z_axis-(np.dot(z_axis, v)/np.dot(v, v))*v
        for f_index in faces:
            if f_index == -1:
                pass
            else:
                face = mesh_faces[f_index]
                def_face = [[0,1,2]]
                tri_vertices = [ mesh_vertex_3d[face[0]], mesh_vertex_3d[face[1]], mesh_vertex_3d[face[2]]]
                triangle = trimesh.Trimesh(vertices=tri_vertices, faces=def_face)
                w = np.cross(plane_normal, v)
                line = mesh_plane(mesh=triangle, plane_normal=w, plane_origin=vertex)
                if line.shape[0]>0:
                    lines.append(line[0])
        vectors = []
        for line in lines:
            v = line-vertex;
            non_zero_v = v[np.nonzero(v)]
            vectors.append(non_zero_v)

        if len(vectors)>0:
            max_neg = 0
            max_pos = 0
            max_ang_neg = 0
            max_ang_pos = 0
            v0 = vectors[0]
            for i in range(1,len(vectors)):
                vi = vectors[i]
                cross = np.dot(np.cross(v0,vi),w)
                dot = abs(np.dot(v0,vi))
                if cross>0 and dot>max_ang_pos:
                    max_pos = i
                    max_ang_pos = dot
                elif cross<0 and dot>max_ang_neg:
                    max_neg = i
                    max_ang_neg = dot
                else:
                    pass
            angle = max_ang_pos+max_ang_neg
            arclen = np.linalg.norm(vectors[max_pos])+np.linalg.norm(vectors[max_neg])

            curvature_tangent_surface.append(angle/arclen)
        else:
            curvature_tangent_surface.append(-1)

curvature = np.asarray(curvature_tangent_surface)
curvature_tangent_surface_final = curvature_tangent_surface[np.argwhere(curvature_tangent_surface>=0 and curvature_contours_surface >=0 )]
curvature_contours_surface_final = curvature_contours_surface[np.argwhere(curvature_tangent_surface>=0 and curvature_contours_surface >=0 )]


plt.show()

cmap_positive = plt.get_cmap("Blues")  # 用于大于0的值
cmap_negative = mcolors.ListedColormap(["red"])  # 用于-1
# 使用Normalize对象规范化颜色数据
norm = mcolors.Normalize(vmin=np.argwhere(curvature>0).min(), vmax=curvature.max(), clip=True)

# 使用plt.plot绘制多边形

plt.plot(x_, y_, 'k-')

# 使用scatter绘制顶点，并根据顶点颜色进行着色
for i in range(len(curvature)):
    if curvature[i] == -1:
        plt.scatter(x_[i], y_[i], color=cmap_negative(norm(curvature[i])))
    else:
        plt.scatter(x_[i], y_[i], color=cmap_positive(norm(curvature[i])))

plt.show()