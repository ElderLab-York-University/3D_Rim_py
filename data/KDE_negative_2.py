import pickle

import seaborn as sns
from sklearn.metrics import r2_score
from statsmodels.nonparametric.kernel_regression import KernelReg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.stats import gaussian_kde, norm
from tqdm import tqdm

# 打开文件，准备读取
with open("Thingi10k_data2.pkl", "rb") as f:
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


normal_curvature = normal_curvature[
    (apparent_curvature != 0) & (apparent_curvature > -1e5) & (apparent_curvature < 0) & (abs(apparent_curvature) > 1e-4)]
apparent_curvature = apparent_curvature[
    (apparent_curvature != 0) & (apparent_curvature > -1e5) & (apparent_curvature < 0) & (abs(apparent_curvature) > 1e-4)]

apparent_curvature = apparent_curvature[normal_curvature <= 1e6]
normal_curvature = normal_curvature[normal_curvature <= 1e6]

apparent_curvature = np.log10(-apparent_curvature)
apparent_curvature = apparent_curvature
df = pd.DataFrame({'apparent_curvature': apparent_curvature, 'normal_curvature': normal_curvature})

# Perform Kernel Regression
model = KernelReg(endog=df['normal_curvature'], exog=df['apparent_curvature'], var_type='c', reg_type='ll', bw=[0.1])
apparent_curvature_test = np.linspace(min(df['apparent_curvature']), max(df['apparent_curvature']), 500)
mean_normal_curvature, mfx = model.fit(apparent_curvature_test)



predicted_normal_curvature = np.array([model.fit([x])[0][0] for x in tqdm(df['apparent_curvature'], desc="Predicting")])
with open("predicted_normal_curvature_thingi10K2N.pkl", "wb") as f:
    # 使用pickle.dump将字典保存到文件
    pickle.dump(predicted_normal_curvature, f)
'''
with open("predicted_normal_curvature_thingi10K2N.pkl", "rb") as f:
    # 使用pickle.load从文件中加载字典
    predicted_normal_curvature = pickle.load(f)
'''
residuals = df['normal_curvature'] - predicted_normal_curvature
se = np.sqrt(np.var(residuals))

explained_variance = 1 - np.var(residuals)/np.var(df['normal_curvature'] )
r2 = r2_score(normal_curvature, predicted_normal_curvature)

print("explained_variance:"+str(explained_variance))
print("R2:"+str(r2))

fig, ax = plt.subplots()
ax.scatter(df['apparent_curvature'], df['normal_curvature'], color='black', label='Data',s=0.1,edgecolors='none')
ax.plot(apparent_curvature_test, mean_normal_curvature, color='blue', label='Mean Prediction')
ax.fill_between(apparent_curvature_test, mean_normal_curvature - se, mean_normal_curvature + se, color='blue',
                 alpha=0.2, label='1 Std. Error')

# 设置y轴的标签格式为10的幂
labels_y = [f'$-10^{int(val)}$' for val in ax.get_yticks()]
ax.set_yticklabels(labels_y)
labels_x = [f'$-10^{int(val)}$' for val in ax.get_xticks()]
ax.set_xticklabels(labels_x)
plt.xlabel("Apparent Curvature")
plt.ylabel("Normal Curvature")
plt.title("Mean and Standard Error of Normal Curvature vs Apparent Curvature")
plt.legend()
plt.show()