import fnmatch
import math
import os
import random
import multiprocessing
from p_tqdm import p_uimap, p_map
import numpy as np
from tqdm import tqdm
from experiments.cal_give_obj import cal
from data import renderObj

shape_net_core_path = '../ShapeNetCore.v2'
sample_path = '../ShapeNetCoreSample2'
max_num_sample_pre_class = 10
max_num_sample_pre_obj = 10
num_class = 55

sampled = np.zeros(55)


def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def find_obj_files(path):
    obj_files = []

    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, '*.obj'):
            obj_files.append(os.path.join(root, filename))

    return obj_files




if __name__ == '__main__':
    class_path = get_subdirectories(shape_net_core_path)

    all_obj_files = []
    print('parser_obj')
    for class_path in tqdm(class_path):
        obj_files = find_obj_files(os.path.join(shape_net_core_path, class_path))
        all_obj_files.append(obj_files)

    sampled_objs = []
    print('sample_obj')
    for obj_files in tqdm(all_obj_files):
        combinations = random.sample(obj_files, min(len(obj_files), max_num_sample_pre_class))
        sampled_objs.extend(combinations)

    with multiprocessing.Pool(processes=8) as pool:
        # 使用 tqdm 显示进度
        for _ in tqdm(pool.imap_unordered(cal, sampled_objs), total=len(sampled_objs)):
            pass

