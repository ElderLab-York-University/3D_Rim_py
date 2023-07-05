import fnmatch
import math
import os
import pickle
import random
import multiprocessing

import matplotlib.pyplot as plt
from p_tqdm import p_uimap, p_map
import numpy as np
from tqdm import tqdm
from experiments.cal_give_obj import cal
from data import renderObj
import re
sample_path = '../ShapeNetCoreSample2'
max_num_sample_pre_class = 10
max_num_sample_pre_obj = 10
num_class = 55

sampled = np.zeros(55)

total_correlation = []

def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def load_npz_files(directory):
    # 正则表达式模式
    pattern = re.compile(r'(\d+)_\d+npz\.npz')

    # 存储文件信息的字典
    file_dict = {}

    # 遍历目录中的所有文件和子目录
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            match = pattern.match(filename)

            # 如果文件名符合模式
            if match:
                a = match.group(1)

                # 如果字典中已经有a这个键
                if a in file_dict:
                    file_dict[a].append(os.path.join(dirpath, filename))
                else:
                    file_dict[a] = [os.path.join(dirpath, filename)]

    # 加载npz文件
    c1 = {}
    c2 = {}
    contours = {}
    rotations = {}
    filenames_refer = {}
    class_based_correlation = {}
    for a in file_dict:
        c1[a] = []
        c2[a] = []
        contours[a] = []
        rotations[a] = []
        class_based_correlation[a] = []
        filenames_refer[a] = []
        for file in file_dict[a]:
            with np.load(file) as data:
                c1[a].append(data['c1'])
                c2[a].append(data['c2'])
                contours[a].append(data['contours'])
                rotations[a].append(data['rotations'])
                filenames_refer[a].append(file)
                correlation_file = np.corrcoef(data['c1'],data['c2'])[0][1]
                class_based_correlation[a].append(correlation_file)
                total_correlation.append(correlation_file)

    return c1,c2,contours,rotations,class_based_correlation,filenames_refer

class_paths = get_subdirectories(sample_path)

all_npz_files = []
print('parser_npz')
for class_path in tqdm(class_paths):
    npz_dict = load_npz_files(os.path.join(sample_path, class_path))
    all_npz_files.append(npz_dict)

with open("total_correlation.pkl", "wb") as f:
    # 使用pickle.dump将字典保存到文件
    pickle.dump(total_correlation, f)

with open("all_npz_files.pkl", "wb") as f:
    # 使用pickle.dump将字典保存到文件
    pickle.dump(all_npz_files, f)
plt.hist(total_correlation,bins=50)
plt.show()
