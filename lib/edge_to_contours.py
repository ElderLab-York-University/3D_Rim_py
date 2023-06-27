import numpy as np
from shapely.geometry import LineString, Point
from pyqtree import Index
import numpy as np


def edge_to_contours(points,edges):
    '''
    :param points: (n,2) points , each line is x,y coord
    :param edges: (m,2) edges, each line is index of two connected points.
    :return:
    '''
    new_points,new_edges = process_edges(points,edges)
    new_points, new_edges = points,edges
    edge_index,point_index = build_edge_index(new_edges)
    minIdx = np.lexsort((new_points[:,0],new_points[:,1]))[0]
    currentIdx = minIdx
    lastAngle = 0
    contours = []
    contours.append(minIdx)
    while(True):
        nextIdx = get_next(lastAngle,new_points,new_edges,edge_index,point_index,currentIdx)
        if nextIdx == minIdx:
            break
        else:
            lastAngle = calculate_angle(points[currentIdx],points[nextIdx])
            currentIdx = nextIdx
            contours.append(nextIdx)
    return contours
def calculate_angle(point1, point2):
    # 创建向量
    vector = np.subtract(point2, point1)

    # 计算向量与x轴的夹角
    angle = np.arctan2(vector[1], vector[0])

    # 将角度转化为度数
    angle = np.degrees(angle)

    # 如果得到的角度是负数，我们可以通过添加360度将其转换为正数
    if angle < 0:
        angle += 360

    return angle

def get_next(lastAngle,points,edges,edge_index,point_index,idx):
    curr_edges = edge_index[idx]
    curr_points = point_index[idx]
    angles = []
    for idx in curr_edges:
        pointidxA,pointidxB = edges[idx]
        angle = calculate_angle(points[pointidxA],points[pointidxB])
        angles.append(angle)
    arr = np.array(angles)
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]
    idx = np.searchsorted(sorted_arr, lastAngle, side='right')
    if idx < len(sorted_arr):
        original_position = sorted_indices[idx]
    else:
        original_position = sorted_indices[0]

    next_point = curr_points[original_position]

    return next_point

def build_edge_index(edges):
    edge_index = {}
    point_index = {}
    # 遍历所有的边
    for i, edge in enumerate(edges):
        # 对于每一条边，将它添加到起点和终点的边列表中
        if edge[0] not in edge_index:
            edge_index[edge[0]] = []
            point_index[edge[0]] = []
        if edge[1] not in edge_index:
            edge_index[edge[1]] = []
            point_index[edge[1]] = []
        edge_index[edge[0]].append(i)
        edge_index[edge[1]].append(i)
        point_index[edge[0]].append(edge[1])
        point_index[edge[1]].append(edge[0])
    return edge_index,point_index
def process_edges(points, edges):
    new_points = points.tolist()
    new_edges = edges.tolist()
    is_new_edge = [False] * len(edges)

    # 初始化四叉树
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    spindex = Index(bbox=(x_min, y_min, x_max, y_max))

    # 将每一条边作为一个项目插入四叉树
    edge_items = []
    for idx, edge in enumerate(edges):
        line = LineString([points[edge[0]], points[edge[1]]])
        edge_items.append((line, idx))
        spindex.insert(item=idx, bbox=line.bounds)

        # 遍历每一个项目，查询可能相交的其他项目
    for line,idx in edge_items:
        candidates = spindex.intersect(bbox=line.bounds)
        for cidx in candidates:
            candline = edge_items[cidx][0]
            if line.intersects(candline):
                intersection = line.intersection(candline)
                if isinstance(intersection, Point):
                    new_points.append(intersection.coords[0])
                    new_point_index = len(new_points) - 1
                    # Replace old edges with new edges
                    spindex.remove(item=idx, bbox=line.bounds)
                    spindex.remove(item=cidx, bbox=candline.bounds)

                    new_edges[idx][0] = new_point_index
                    new_edges[cidx][0] = new_point_index
                    new_edges.append([edges[idx][1], new_point_index])
                    new_edges.append([edges[cidx][1], new_point_index])
                    # Insert new edges into quadtree
                    new_edge1 = LineString([points[edges[idx][0]], intersection.coords[0]])
                    edge_items[idx] = (new_edge1, idx)
                    spindex.insert(item=idx, bbox=new_edge1.bounds)

                    new_edge2 = LineString([points[edges[cidx][0]], intersection.coords[0]])
                    edge_items[cidx] = (new_edge2, cidx)
                    spindex.insert(item=cidx, bbox=new_edge2.bounds)

                    new_edge3 = LineString([points[edges[idx][1]], intersection.coords[0]])
                    edge_items.append((new_edge3, len(edge_items)))
                    spindex.insert(item=len(edge_items)-1, bbox=new_edge3.bounds)

                    new_edge4 = LineString([points[edges[cidx][1]], intersection.coords[0]])
                    edge_items.append((new_edge4, len(edge_items)))
                    spindex.insert(item=len(edge_items)-1, bbox=new_edge4.bounds)
                    # Add True to is_new_edge for new edges
                    is_new_edge[idx] = True;
                    is_new_edge[cidx] = True;
                    is_new_edge.append(True)
                    is_new_edge.append(True)
    '''
    for idx,edge in enumerate(edges):
        line = LineString([points[edge[0]], points[edge[1]]])
        edge_items.append((line,edge,idx))
        spindex.insert(item=edge_items[-1], bbox=line.bounds)

    # 遍历每一个项目，查询可能相交的其他项目
    for iidx,edge_item in enumerate(edge_items):
        line, edge, edge_idx = edge_item
        candidates = spindex.intersect(bbox=line.bounds)
        for cidx,candidates_item in enumerate(candidates):
            candidate_line, candidate_edge, candidate_idx = candidates_item
            if line.intersects(candidate_line):
                intersection = line.intersection(candidate_line)
                if isinstance(intersection, Point):
                    new_points.append(intersection.coords[0])
                    new_point_index = len(new_points) - 1
                    # Replace old edges with new edges
                    spindex.remove(item=edge_items[edge_idx],bbox=line.bounds)
                    spindex.remove(item=edge_items[candidate_idx],bbox=candidate_line.bounds)

                    new_edges[edge_idx][0] = new_point_index
                    new_edges[candidate_idx][0] = new_point_index
                    new_edges.append([edge[1], new_point_index])
                    new_edges.append([candidate_edge[1], new_point_index])
                    # Insert new edges into quadtree
                    new_edge1 = LineString([points[edge[0]], intersection.coords[0]])
                    edge_items[edge_idx] =(new_edge1, new_edges[edge_idx],edge_idx)
                    spindex.insert(item=edge_items[edge_idx], bbox=new_edge1.bounds)

                    new_edge2 = LineString([points[candidate_edge[0]], intersection.coords[0]])
                    edge_items[candidate_idx] =(new_edge2, new_edges[candidate_idx],candidate_idx)
                    spindex.insert(item=edge_items[candidate_idx] , bbox=new_edge2.bounds)

                    new_edge3 = LineString([points[edge[1]], intersection.coords[0]])
                    edge_items.append((new_edge3, new_edges[-1],len(new_edges)-2))
                    spindex.insert(item=edge_items[-1], bbox=new_edge3.bounds)

                    new_edge4 = LineString([points[candidate_edge[1]], intersection.coords[0]])
                    edge_items.append((new_edge4, new_edges[-3],len(new_edges)-1))
                    spindex.insert(item=edge_items[-1], bbox=new_edge4.bounds)
                    # Add True to is_new_edge for new edges
                    is_new_edge[edge_idx] = True;
                    is_new_edge[candidate_idx] = True;
                    is_new_edge.append(True)
                    is_new_edge.append(True)
    '''

    return np.array(new_points), np.array(new_edges), np.array(is_new_edge)


