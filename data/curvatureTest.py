import numpy as np
from genContours import genSampledContours
from  cal2DCurvature import  compute_2dcurvatures
from  cal3DCurvature import  compute_3dcurvatures
data = np.load('depth.npz')
depth = data['depth']
height, width = depth.shape[:2]
center_x = width // 2
center_y = height // 2
Contours = genSampledContours(depth,128)
print('finish sample')
if len(Contours)>1:
    pass
else:
    contour = Contours[0]
    print(len(contour))
    curveture2d = compute_2dcurvatures(contour)
    print('finish 2d')
    curveture3d = compute_3dcurvatures(depth,contour,(center_x, center_y))
print(curveture2d)
print(curveture3d)
print(np.corrcoef(curveture3d,curveture2d))
print(len(curveture3d))
print('finished')