import os
import threading
import itertools
import random
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

contours = []
correlations = []

def getImageFiles(directory):
    imageFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                imageFiles.append(os.path.join(root, file))
    return imageFiles

def compareContours(image1,image2):
    _, thresh1 = cv2.threshold(image1, 1, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(image2, 1, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours1 or not  contours2:
        return None
    # 取最大轮廓
    contour1 = max(contours1, key=cv2.contourArea)
    contour2 = max(contours2, key=cv2.contourArea)

    # 使用matchShapes函数比较轮廓
    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)

    return abs(similarity)+1

def calculate_correlation(image1, image2):
    # 将图像转换为浮点数并归一化
    image1_norm = image1.astype(np.float32) / 255.0
    image2_norm = image2.astype(np.float32) / 255.0

    # 计算图像的平均值和标准差
    mean1, std_dev1 = cv2.meanStdDev(image1_norm)
    mean2, std_dev2 = cv2.meanStdDev(image2_norm)

    # 检查除数是否为零或接近零
    eps = 1e-8
    if std_dev1 * std_dev2 < eps:
        return None
    else:
        covariance = cv2.mean((image1_norm - mean1) * (image2_norm - mean2))[0]
        correlation_coefficient = covariance / (std_dev1 * std_dev2)
    return abs(correlation_coefficient)

def process_image_pair(image1, image2 ):
    cvalue = compareContours(image1, image2)
    cvalue2 = calculate_correlation(image1, image2)
    if cvalue is not None and cvalue2 is not None:
        contours.append(cvalue)
        correlations.append(cvalue2)

dir = "../ShapeNetCore_Depth"
imageFiles = getImageFiles(dir)

images = []
print("filecounter:"+str(len(imageFiles)))
for imageFile in imageFiles:
    images.append(cv2.imread(imageFile,cv2.IMREAD_GRAYSCALE))

print("finsih reading")
k = 20000
all_image_pairs = list(itertools.combinations(range(len(images)), 2))
selected_image_pairs = random.sample(all_image_pairs, k)
print("start")

for pair in tqdm(selected_image_pairs):
    idx1,idx2 = pair
    process_image_pair(images[idx1],images[idx2])


plt.hexbin(contours, correlations, gridsize=100, cmap='viridis',xscale="log")
plt.colorbar()
plt.xlabel('contours Hu moment values')
plt.ylabel('depthImage pearson correlation coefficient')
plt.title('Scatter Plot of contours and depthImage correlation')
plt.show()
plt.savefig("./result/r2.png", dpi=200)