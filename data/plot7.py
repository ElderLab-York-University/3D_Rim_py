import pickle
from statsmodels.nonparametric.kernel_regression import KernelReg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.stats import gaussian_kde, norm
from tqdm import tqdm

# 打开文件，准备读取
with open("all_npz_files_v3.pkl", "rb") as f:
    # 使用pickle.load从文件中加载字典
    all_npz_files = pickle.load(f)
apparent_curvature = []
normal_curvature = []
counter = []
contour_counter = []
angles = []
for i, obj_class in enumerate(all_npz_files):
    c1, c2, contours, rotations, class_based_correlation, filenames_refer, validmaps, angle = obj_class
    flattened_array = np.concatenate(c1['4'])

    c1_normalized = []
    c2_normalized = []
    for i, apparent in enumerate(c1['4']):
        normal = c2['4'][i]
        local_c = validmaps['4'][i]
        local_angle = angle['4'][i]
        local_apparent = apparent
        contour_counter.append(len(local_c))
        counter.append(apparent.size)
        angles.extend(local_angle)
        c1_normalized.extend(local_apparent)
        c2_normalized.extend(normal)

    apparent_curvature.extend(c1_normalized)
    normal_curvature.extend(c2_normalized)

normal_curvature = np.asarray(normal_curvature)
apparent_curvature = np.asarray(apparent_curvature)
normal_curvature = normal_curvature[
    (apparent_curvature != 0) & (apparent_curvature >= -1) & (apparent_curvature <= 1) & (abs(apparent_curvature) > 1e-7)]
apparent_curvature = apparent_curvature[
    (apparent_curvature != 0) & (apparent_curvature >= -1) & (apparent_curvature <= 1) & (abs(apparent_curvature) > 1e-7)]

apparent_curvature = apparent_curvature[normal_curvature <= 10000]
normal_curvature = normal_curvature[normal_curvature <= 10000]
# plt.scatter(apparent_curvature,normal_curvature,s=0.02)
'''
print(sum(contour_counter))
print(sum(counter))
plt.xlim([-1,1])
plt.ylim([0,10000])

plt.ylabel('apparent curvature probability p(γ_P)')
plt.xlabel('apparent curvature magnitude |γ_P| (radian/length)')
plt.title('distribution of apparent curvature (magnitude|linear scale)')
plt.show()
'''

df = pd.DataFrame({'apparent_curvature': apparent_curvature, 'normal_curvature': normal_curvature})
print(1)
# Perform Kernel Regression
model = KernelReg(endog=df['normal_curvature'], exog=df['apparent_curvature'], var_type='c', reg_type='ll', bw=[0.1])
apparent_curvature_test = np.linspace(min(df['apparent_curvature']), max(df['apparent_curvature']), 1000)
mean_normal_curvature, mfx = model.fit(apparent_curvature_test)

print(2)
# Compute standard error
predicted_normal_curvature = np.array([model.fit([x])[0][0] for x in tqdm(df['apparent_curvature'], desc="Predicting")])
residuals = df['normal_curvature'] - predicted_normal_curvature
se = np.sqrt(np.var(residuals))
print(3)
# Plotting
with open("mean_normal_curvature.pkl", "wb") as f:
    # 使用pickle.dump将字典保存到文件
    pickle.dump(mean_normal_curvature, f)

with open("se.pkl", "wb") as f:
    # 使用pickle.dump将字典保存到文件
    pickle.dump(se, f)
'''
with open("se.pkl", "rb") as f:
    # 使用pickle.load从文件中加载字典
    se = pickle.load(f)
'''
plt.figure(figsize=(10, 6))
plt.scatter(df['apparent_curvature'], df['normal_curvature'], color='black', label='Data',s=0.1,edgecolors='none')
plt.plot(apparent_curvature_test, mean_normal_curvature, color='blue', label='Mean Prediction')
plt.fill_between(apparent_curvature_test, mean_normal_curvature - se, mean_normal_curvature + se, color='blue',
                 alpha=0.2, label='1 Std. Error')
plt.xlabel("Apparent Curvature")
plt.ylabel("Normal Curvature")
#plt.xscale('symlog',linthresh=1e-5)
plt.title("Mean and Standard Error of Normal Curvature vs Apparent Curvature|bandwith=0.1")
plt.legend()
plt.show()

'''
plt.gcf().set_dpi(600)
plt.scatter(apparent_curvature,normal_curvature,s=0.1,edgecolors='none')
plt.xscale('log')
plt.yscale('log')

plt.title('scatter plot of apparent curvature vs normal curvature \n (logscale|apparent curvature normalized by number of point in the contours|45 degree fov)')
plt.show()
'''
