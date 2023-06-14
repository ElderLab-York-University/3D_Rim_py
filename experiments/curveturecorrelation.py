import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from compareCurvature import compare_curvature
def get_npz_files(directory):
    npz_files = []

    # 使用 os.walk 遍历目录
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # 如果文件是 .npz 文件，则将其添加到列表中
            if filename.endswith('.npz'):
                npz_files.append(os.path.join(dirpath, filename))

    return npz_files

corrections = []

directory = '../ShapeNetCore_100_npz/fov2'  # 替换为你的目录
npz_files = get_npz_files(directory)

for file in tqdm(npz_files):
    local_correction=compare_curvature(file)
    corrections.append(local_correction)

print('finished')
np.save('corrections.npy',corrections)
plt.plot(corrections)