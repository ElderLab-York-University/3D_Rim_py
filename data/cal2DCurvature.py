import numpy as np

def compute_2dcurvatures(contours):
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
        prev = contours[(i-1)%n]
        curr = contours[i]
        next = contours[(i+1)%n]

        # 计算两条线段的长度
        L1 = np.linalg.norm(curr - prev)
        L2 = np.linalg.norm(next - curr)
        L = (L1 + L2) / 2.0

        # 计算两条线段的夹角
        vector1 = prev - curr
        vector2 = next - curr
        cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        theta = np.arccos(np.clip(cos_theta, -1, 1))  # 限制在 [-1, 1] 以避免浮点误差

        # 计算曲率
        curvatures[i] = theta / L

    return curvatures