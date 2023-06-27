
import numpy as np
from shapely import Polygon, union_all


def projected(mesh,
              normal,
              matrix,
              **kwargs):
    """
    Project a mesh onto a plane and then extract the
    polygon that outlines the mesh projection on that
    plane.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Source geometry
    check : bool
      If True make sure is flat
    normal : (3,) float
      Normal to extract flat pattern along
    origin : None or (3,) float
      Origin of plane to project mesh onto
    pad : float
      Proportion to pad polygons by before unioning
      and then de-padding result by to avoid zero-width gaps.
    tol_dot : float
      Tolerance for discarding on-edge triangles.
    max_regions : int
      Raise an exception if the mesh has more than this
      number of disconnected regions to fail quickly before unioning.

    Returns
    ----------
    projected : trimesh.path.Path2D
      Outline of source mesh
    """
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