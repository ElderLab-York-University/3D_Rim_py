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
sample_path = '../Thingi10KSample4_2'
max_num_sample_pre_class = 10
max_num_sample_pre_obj = 10
num_class = 55

sampled = np.zeros(55)

total_correlation = {}

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
    valid_maps = {}
    angle = {}
    vertex_3d = {}
    for a in file_dict:
        if a not in total_correlation:
            total_correlation[a] = []
        c1[a] = []
        c2[a] = []
        contours[a] = []
        rotations[a] = []
        filenames_refer[a] = []
        valid_maps[a] = []
        angle[a] = []
        vertex_3d[a] = []
        for file in file_dict[a]:
            with np.load(file,allow_pickle=True) as data:
                vaild_map = data['valid_map']
                counter_ = data['contours']
                rotation_ = data['rotations']
                c1_ = data['c1']
                c2_ = data['c2']
                angle_ = data['angle']
                vertex_3d_ = data['vertex_3d']

                c1_ = c1_[vaild_map]
                c2_ = c2_[vaild_map]
                angle_ = angle_[vaild_map]

                vertex_3d[a].append(vertex_3d_)
                angle[a].append(angle_)
                c1[a].append(c1_)
                c2[a].append(c2_)
                contours[a].append(counter_)
                rotations[a].append(rotation_)
                filenames_refer[a].append(file)
                valid_maps[a].append(vaild_map)

    return c1,c2,contours,rotations,filenames_refer,valid_maps,angle

class_paths = get_subdirectories(sample_path)

all_npz_files = []
print('parser_npz')
npz_dict = load_npz_files(sample_path)

with open("Thingi10k_data4_2.pkl", "wb") as f:
    # 使用pickle.dump将字典保存到文件
    pickle.dump(npz_dict, f)

