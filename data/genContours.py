import cv2
import numpy as np
from data.sampleContours import sampleContour

def genSampledContours(depth,num=128):
    binary_img = np.where(depth > 0, 1, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sampledContours = []
    for contour in contours:
        reshaped = contour.squeeze(axis=1)
        sampled = sampleContour(reshaped,num)
        sampledContours.append(sampled)
    return sampledContours