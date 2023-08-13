import pickle
from statsmodels.nonparametric.kernel_regression import KernelReg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.stats import gaussian_kde, norm
from tqdm import tqdm

# 打开文件，准备读取
with open("Thingi10k_data4.pkl", "rb") as f:
    # 使用pickle.load从文件中加载字典
    all_npz_files = pickle.load(f)
apparent_curvature = []
normal_curvature = []
counter = []
contour_counter = []
angles = []
c1, c2, contours, rotations, filenames_refer,validmaps,angle = all_npz_files
flattened_array = np.concatenate(c1['4'])

for i,apparent in enumerate(c1['4']):
    normal = c2['4'][i]
    local_c = validmaps['4'][i]
    local_angle = angle['4'][i]
    local_apparent = apparent
    contour_counter.append(len(local_c))
    counter.append(apparent.size)
    angles.extend(local_angle)
    apparent_curvature.extend(local_apparent)
    normal_curvature.extend(normal)




normal_curvature = np.asarray(normal_curvature)
apparent_curvature = np.asarray(apparent_curvature)
angles = np.asarray(angles)

boolarray = (apparent_curvature != 0) & (abs(apparent_curvature)>1e-4)
normal_curvature = normal_curvature[
    boolarray ]
#angles = angles[ boolarray]
apparent_curvature = apparent_curvature[
    boolarray]

#angles = angles[normal_curvature <= 10000]
apparent_curvature = apparent_curvature[(normal_curvature <= 1e6) & (normal_curvature !=0)]
normal_curvature = normal_curvature[(normal_curvature <= 1e6) & (normal_curvature != 0)]


plt.scatter(apparent_curvature,normal_curvature,s=0.1,edgecolors='None')
corr_matrix = np.corrcoef(apparent_curvature, normal_curvature)
corr_xy = corr_matrix[0, 1]
plt.xscale('symlog',linthresh=1e-1)
print(corr_xy)
plt.yscale('log')
print(sum(contour_counter))
print(sum(counter))

plt.xlabel("Apparent Curvature")
plt.ylabel("Normal Curvature")
plt.title('Scatter plot of apparent curvature vs normal curvature')

plt.show()





'''
plt.gcf().set_dpi(600)
plt.scatter(apparent_curvature,normal_curvature,s=0.1,edgecolors='none')
plt.xscale('log')
plt.yscale('log')

plt.title('scatter plot of apparent curvature vs normal curvature \n (logscale|apparent curvature normalized by number of point in the contours|45 degree fov)')
plt.show()
'''