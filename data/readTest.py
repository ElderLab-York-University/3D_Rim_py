import os
import fnmatch

def find_obj_files(path):
    obj_files = []

    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, '*.obj'):
            obj_files.append(os.path.join(root, filename))

    return obj_files

path = '../ShapeNetCore.v2'
obj_files = find_obj_files(path)
