import fnmatch
import math
import os
import pickle
import random
import multiprocessing
from tqdm import tqdm
from p_tqdm import p_uimap, p_map
import numpy as np
from experiments.cal_give_obj import cal
from data import renderObj
from experiments.cal_give_obj2 import cal2

Thingi10K_path = '../Thingi10K/Models'
max_num_sample_pre_class = 200


def get_subdirectories(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def find_stl_files(path):
    obj_files = []

    for root, dirs, files in os.walk(path):
        for pattern in ['*.stl', '*.STL']:
            for filename in fnmatch.filter(files, pattern):
                obj_files.append(os.path.join(root, filename))

    return obj_files



if __name__ == '__main__':
    '''
    class_path = get_subdirectories(Thingi10K_path)

    all_obj_files = []
    print('parser_stl')
    stl_files = find_stl_files(Thingi10K_path)
    sampled_stls = []
    print('sample_stl')
    combinations = random.sample(stl_files, min(len(stl_files), max_num_sample_pre_class))
    sampled_stls.extend(combinations)
   
    with open("sampled_stls.pkl", "wb") as f:
        # 使用pickle.dump将字典保存到文件
        pickle.dump(sampled_stls, f)
    '''
    with open("sampled_stls.pkl", "rb") as f:
        # 使用pickle.load从文件中加载字典
        sampled_stls = pickle.load(f)

    with multiprocessing.Pool(processes=8) as pool:
        # 使用 tqdm 显示进度
        for _ in tqdm(pool.imap_unordered(cal2, sampled_stls), total=len(sampled_stls)):
            pass

    '''
    for obj in sampled_stls:
        cal2(obj)
    '''
