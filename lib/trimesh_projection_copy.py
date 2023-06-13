
import numpy as np
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
              normal,
              K,RT,
              origin=None,
              ignore_sign=True,
              rpad=1e-5,
              apad=None,
              tol_dot=0.01,
              max_regions=200):
    """
    Project a mesh onto a plane and then extract the polygon
    that outlines the mesh projection on that plane.

    Note that this will ignore back-faces, which is only
    relevant if the source mesh isn't watertight.

    Also padding: this generates a result by unioning the
    polygons of multiple connected regions, which requires
    the polygons be padded by a distance so that a polygon
    union produces a single coherent result. This distance
    is calculated as: `apad + (rpad * scale)`

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
    ignore_sign : bool
      Allow a projection from the normal vector in
      either direction: this provides a substantial speedup
      on watertight meshes where the direction is irrelevant
      but if you have a triangle soup and want to discard
      backfaces you should set this to False.
    rpad : float
      Proportion to pad polygons by before unioning
      and then de-padding result by to avoid zero-width gaps.
    apad : float
      Absolute padding to pad polygons by before unioning
      and then de-padding result by to avoid zero-width gaps.
    tol_dot : float
      Tolerance for discarding on-edge triangles.
    max_regions : int
      Raise an exception if the mesh has more than this
      number of disconnected regions to fail quickly before
      unioning.

    Returns
    ----------
    projected : shapely.geometry.Polygon or None
      Outline of source mesh

    Raises
    ---------
    ValueError
      If max_regions is exceeded
    """
    import numpy as np

    from shapely import ops
    from trimesh import graph
    from trimesh import geometry
    from trimesh import grouping
    from trimesh.transformations import transform_points
    from trimesh.path.polygons import edges_to_polygons
    try:
        import networkx as nx
    except BaseException as E:
        # create a dummy module which will raise the ImportError
        # or other exception only when someone tries to use networkx
        from trimesh.exceptions import ExceptionWrapper
        nx = ExceptionWrapper(E)
    try:
        from rtree import Rtree
    except BaseException as E:
        # create a dummy module which will raise the ImportError
        from trimesh.exceptions import ExceptionWrapper
        Rtree = ExceptionWrapper(E)

        # make sure normal is a unitized copy
    normal = np.array(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # the projection of each face normal onto facet normal
    dot_face = np.dot(normal, mesh.face_normals.T)
    if ignore_sign:
        # for watertight mesh speed up projection by handling side with less faces
        # check if face lies on front or back of normal
        front = dot_face > tol_dot
        back = dot_face < -tol_dot
        # divide the mesh into front facing section and back facing parts
        # and discard the faces perpendicular to the axis.
        # since we are doing a unary_union later we can use the front *or*
        # the back so we use which ever one has fewer triangles
        # we want the largest nonzero group
        count = np.array([front.sum(), back.sum()])
        if count.min() == 0:
            # if one of the sides has zero faces we need the other
            pick = count.argmax()
        else:
            # otherwise use the normal direction with the fewest faces
            pick = count.argmin()
        # use the picked side
        side = [front, back][pick]
    else:
        # if explicitly asked to care about the sign
        # only handle the front side of normal
        side = dot_face > tol_dot

    # subset the adjacency pairs to ones which have both faces included
    # on the side we are currently looking at
    adjacency_check = side[mesh.face_adjacency].all(axis=1)
    adjacency = mesh.face_adjacency[adjacency_check]

    # a sequence of face indexes that are connected
    face_groups = graph.connected_components(
        adjacency, nodes=np.nonzero(side)[0])

    # if something is goofy we may end up with thousands of
    # regions that do nothing except hang for an hour then segfault
    if len(face_groups) > max_regions:
        raise ValueError('too many disconnected groups!')

    # reshape edges into shape length of faces for indexing
    edges = mesh.edges_sorted.reshape((-1, 6))
    # transform from the mesh frame in 3D to the XY plane
    to_2D = geometry.plane_transform(
        origin=origin, normal=normal)
    # transform mesh vertices to 2D and clip the zero Z

    vertices_2D = map_points_to_image(mesh.vertices,K,RT)
    print(vertices_2D.shape)


    polygons = []
    for faces in face_groups:
        # index edges by face then shape back to individual edges
        edge = edges[faces].reshape((-1, 2))
        # edges that occur only once are on the boundary
        group = grouping.group_rows(edge, require_count=1)
        # turn each region into polygons
        polygons.extend(edges_to_polygons(
            edges=edge[group], vertices=vertices_2D))

    padding = 0.0
    if apad is not None:
        # set padding by absolute value
        padding += float(apad)
    if rpad is not None:
        # get the 2D scale as the longest side of the AABB
        scale = vertices_2D.ptp(axis=0).max()
        # apply the scale-relative padding
        padding += float(rpad) * scale

    # some types of errors will lead to a bajillion disconnected
    # regions and the union will take forever to fail
    # so exit here early
    if len(polygons) > max_regions:
        raise ValueError('too many disconnected groups!')

    # if there is only one region we don't need to run a union
    elif len(polygons) == 1:
        return polygons[0]
    elif len(polygons) == 0:
        return None
    else:
        # inflate each polygon before unioning to remove zero-size
        # gaps then deflate the result after unioning by the same amount
        # note the following provides a 25% speedup but needs
        # more testing to see if it deflates to a decent looking
        # result:
        # polygon = ops.unary_union(
        #    [p.buffer(padding,
        #              join_style=2,
        #              mitre_limit=1.5)
        #     for p in polygons]).buffer(-padding)
        polygon = ops.unary_union(
            [p.buffer(padding)
             for p in polygons]).buffer(-padding)
    return polygon
def map_points_to_image(points_3d, K, RT):
    # 把点从世界坐标系变换到相机坐标系
    points_cam = points_3d @ RT[:, :3].T + RT[:, 3]

    # 把点从相机坐标系映射到图像平面
    points_2d = points_cam @ K.T

    # 进行透视除法，使得点的坐标是归一化的齐次坐标
    points_2d = points_2d / points_2d[:, 2, np.newaxis]

    return points_2d[:, :2]  # 只返回x和y坐标