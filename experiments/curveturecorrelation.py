import os
import random

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

directory = '../ShapeNetCore_100_npz/fov2'
k = 250
npz_files = get_npz_files(directory)
selected_files = random.sample(npz_files
, k)
for i in range(10,100,20):
    for file in tqdm(selected_files):
        local_correction=compare_curvature(file,i)
        corrections.append(local_correction)
    np.save('correctionsfov2'+str(i)+'.npy', corrections)
    plt.hist(corrections, bins=50)
    plt.show()
    plt.savefig('correctionsfov2' + str(i) + ".png")

print('finished')
