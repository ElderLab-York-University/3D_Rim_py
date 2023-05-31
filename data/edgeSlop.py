import cv2
import numpy as np
from scipy.spatial import KDTree


def create_contour_line_segments(image):
    # 找到边缘A
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours_A, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 创建一个空的线段列表
    line_segments = []

    # 创建一个空的点列表
    contour_points = []

    # 遍历边缘A，构建线段
    for contour in contours_A:
        for i in range(len(contour) - 1):
            line_segments.append((tuple(contour[i][0]), tuple(contour[i + 1][0])))
            contour_points.append(contour[i][0])

    # 将点列表转换为数组
    contour_points = np.array(contour_points, dtype=np.float32)
    print(type(contour_points))

    return line_segments, KDTree(contour_points)


def calculate_nearest_line_vectors(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建一个空的RGB图像来保存向量
    vector_image = np.zeros((*gray.shape, 3), dtype=np.uint8)

    # 从灰度图像创建边缘A的线段
    line_segments, tree = create_contour_line_segments(gray)

    # 创建一个副本并删除边缘A
    gray_without_edges_A = np.copy(gray)
    for point in tree.data:
        gray_without_edges_A[point[0], point[1]] = 0

    # 在删除边缘A后找到边缘B
    _, binary = cv2.threshold(gray_without_edges_A, 128, 255, cv2.THRESH_BINARY)
    contours_B, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 对于边缘B上的每一个像素点
    for contour in contours_B:
        for point_B in contour:
            point_B = tuple(point_B[0])
            _, index = tree.query(point_B)
            point_A = tree.data[index]

            # 找到边缘A上与point_A相连的线段
            adjacent_segments = [segment for segment in line_segments if point_A in segment]

            min_vector = None
            min_length = float('inf')

            # 对于每个相邻线段计算垂线向量
            for segment in adjacent_segments:
                point_A1, point_A2 = segment
                vector_AB = np.array(point_B) - np.array(point_A)
                vector_A1A2 = np.array(point_A2) - np.array(point_A1)
                vector = vector_AB - vector_A1A2 * (vector_AB.dot(vector_A1A2) / np.linalg.norm(vector_A1A2) ** 2)

                # 计算向量长度
                length = np.linalg.norm(vector)

                # 更新最短向量
                if length < min_length:
                    min_length = length
                    min_vector = vector

            # 归一化最短向量
            min_vector = min_vector / min_length

            # 保存到vector_image
            vector_image[point_B[0], point_B[1], :2] = (min_vector * 127 + 128).astype(np.uint8)

    # 返回vector_image
    return vector_image


