import math

def calculate_angle(pointA, pointB, pointC, pointD):
    # 解析出坐标值
    x1, y1 = pointA
    x2, y2 = pointB
    x3, y3 = pointC
    x4, y4 = pointD

    # 计算两个向量：AB = (x2-x1, y2-y1) 和 CD = (x4-x3, y4-y3)
    AB = (x2 - x1, y2 - y1)
    CD = (x4 - x3, y4 - y3)

    # 计算两个向量的点积和模的乘积
    dot_product = AB[0] * CD[0] + AB[1] * CD[1]
    mod_product = math.sqrt(AB[0]**2 + AB[1]**2) * math.sqrt(CD[0]**2 + CD[1]**2)

    # 如果模的乘积为0（也就是说，至少有一个向量的长度为0），那么角度未定义，我们返回 None
    if mod_product == 0:
        return None

    # 计算夹角的余弦值
    cos_angle = dot_product / mod_product

    # 由于浮点运算的精度问题，有可能计算得到的余弦值略微超出[-1, 1]的范围，所以我们需要将它截断到[-1, 1]内
    cos_angle = max(min(cos_angle, 1), -1)

    # 计算夹角（弧度）
    angle_rad = math.acos(cos_angle)

    # 将夹角从弧度转换为度
    angle_deg = math.degrees(angle_rad)

    # 返回夹角（度）
    return angle_deg
