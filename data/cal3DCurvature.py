import matplotlib.pyplot as plt
import numpy as np

def bresenham_float(A, B):
    """
    使用Bresenham算法，对两点之间的线段进行采样，返回所有通过的像素坐标。

    参数：
    A : tuple, shape (2,)
        点A的坐标。
    B : tuple, shape (2,)
        点B的坐标。

    返回：
    numpy.array, shape (n, 2)
        所有通过的像素坐标。
    """
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    d = np.round(B - A).astype(np.int32)
    abs_d = np.abs(d)

    if abs_d[0] > abs_d[1]:
        signs = np.sign(d)
        p = abs_d * np.array([2, 1]) - abs_d[0]
        incE = abs_d * np.array([2, 0])
        incNE = abs_d * 2

        points = [np.round(A).astype(np.int32)]
        for _ in range(abs_d[0]):
            if p[0] < 0:
                A[0] += signs[0]
                p += incE
            else:
                A += signs
                p += incNE
            points.append(np.round(A).astype(np.int32))
    else:
        signs = np.sign(d) * np.array([1, 2])
        p = abs_d * np.array([1, 2]) - abs_d[1]
        incE = abs_d * np.array([0, 2])
        incNE = abs_d * np.array([2, 2])

        points = [np.round(A).astype(np.int32)]
        for _ in range(abs_d[1]):
            if p[1] < 0:
                A[1] += signs[1]
                p += incE
            else:
                A += signs
                p += incNE
            points.append(np.round(A).astype(np.int32))

    return np.array(points)
def get_projection_and_distance(A, B, P):
    Ax, Ay = A
    Bx, By = B
    Px, Py = P
    AB_squared = (Bx - Ax)**2 + (By - Ay)**2
    AP_AB = (Px - Ax)*(Bx - Ax) + (Py - Ay)*(By - Ay)
    t = AP_AB / AB_squared
    projection = (Ax + t*(Bx - Ax), Ay + t*(By - Ay))
    distance = ((projection[0]-Ax)**2 + (projection[1]-Ay)**2)**0.5
    return projection, distance

def get_depths(depth_image, points,A,B):
    depths = []
    arcLen = []
    for point in points:
        x, y = point
        if x>=depth_image.shape[0] or y>=depth_image.shape[1] or x<0 or y<0:
            return np.array(depths).reshape(-1),np.array(arcLen).reshape(-1)
        depth = depth_image[y, x]
        if depth > 0 :
            depths.append(depth)
            projection, distance = get_projection_and_distance(A, B, point)
            arcLen.append(distance)
    return np.array(depths).reshape(-1),np.array(arcLen).reshape(-1)

def get_line_pixels_and_projections(img, curr, bisector_normalized):
    posP = curr+bisector_normalized
    negP = curr-bisector_normalized
    points1 = bresenham_float(curr,posP)
    points2 = bresenham_float(curr,negP)

    y,x = get_depths(img,points1,curr,posP)
    y1,x1 = get_depths(img,points2,curr,negP)
    if len(y) > len(y1):
        return x1,y1
    else:
        return x,y

def get_curvature(x, y):
    if(len(x)<3):
        return 0
    coefficients = np.polyfit(x, y, 2)
    a = coefficients[0]
    b = coefficients[1]
    x_mean = np.mean(x)
    curvature = abs(2*a) / (1 + (2*a*x_mean + b)**2)**(3/2)
    return curvature

def get_line_pixels_and_curvature(img, curr, bisector_normalized):

    x,y = get_line_pixels_and_projections(img, curr, bisector_normalized)
    if(len(x)==0):
        return 0
    min_val = np.min(x)
    max_val = np.max(x)
    normalized_x = (x - min_val) / (max_val - min_val)
    curvature = get_curvature(normalized_x, y)
    return curvature
'''
test code
'''
def compute_3dcurvatures(img,contours,factor=40):

    """
       计算每个点的曲率。

       参数：
       contours : numpy.array, shape (n, 2)
           轮廓点坐标列表。

       返回：
       numpy.array, shape (n,)
           每个点的曲率。
       """
    n = contours.shape[0]
    curvatures = np.zeros(n)

    for i in range(n):
        # 取前后两个点，注意处理边界条件
        prev = contours[(i - 1) % n]
        curr = contours[i]
        next = contours[(i + 1) % n]

        # 计算两条线段的长度
        BC = curr - prev
        BA = next - curr


        BA_normalized = BA / np.linalg.norm(BA)
        BC_normalized = BC / np.linalg.norm(BC)

        bisector = BA_normalized + BC_normalized
        bisector_normalized = (bisector / np.linalg.norm(bisector)) * factor

        # 计算曲率
        curvatures[i] = get_line_pixels_and_curvature(img, curr, bisector_normalized)

    return curvatures


