import configparser
import fnmatch
import os
import random
import multiprocessing
from tqdm import tqdm
from experiments.cal_give_obj2 import cal2

config = configparser.ConfigParser()
config.read('../config.ini')
Thingi10K_path = config['SETTING']['DataPath']
npzOutPath = config['SETTING']['NpzOutPath']
NumOfObject = (int)(config['SETTING']['NumOfObject'])
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


    class_path = get_subdirectories(Thingi10K_path)

    all_obj_files = []
    print('parser_stl')
    stl_files = find_stl_files(Thingi10K_path)
    sampled_stls = []
    print('sample_stl')
    combinations = random.sample(stl_files, min(len(stl_files), NumOfObject))
    sampled_stls.extend(combinations)


    with multiprocessing.Pool(processes=8) as pool:
        # 使用 tqdm 显示进度
        for _ in tqdm(pool.imap_unordered(cal2, sampled_stls), total=len(sampled_stls)):
            pass

