# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:42:34 2022

@author: LIMK
"""
import os
import tarfile, zipfile
import pathlib
#import rarfile
'''
def extract(tar_path, target_path):
    try:
        tar = tarfile.open(tar_path, "r")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, dst_dir)
            src_file, dst_file = os.path.join(dst_dir, file_name), os.path.join(dst_dir, pathlib.Path(file_name[:-4]))
            print('extract tar files from:', src_file, 'to:', dst_file)
            if os.path.isdir(dst_file):
                pass
            else:
                os.mkdir(os.path.join(dst_dir, pathlib.Path(file_name[:-4])))
            src_tar = tarfile.open(src_file, "r")
            src_tar.extractall(dst_file)
            src_tar.close()
            os.remove(src_file)
        tar.close()
    except Exception  as e:
        print(e)
'''




def extract(tar_path, target_path):
    try:
        for file_name in os.listdir(tar_path):
            src_file, dst_file = os.path.join(tar_path, file_name), os.path.join(dst_dir, pathlib.Path(file_name[:-4]))
            print('extract tar files from:', src_file, 'to:', dst_file)
            if os.path.isdir(dst_file):
                pass
            else:
                os.mkdir(os.path.join(dst_dir, pathlib.Path(file_name[:-4])))
            src_tar = tarfile.open(src_file, "r")
            src_tar.extractall(dst_file)
            src_tar.close()
    except Exception  as e:
        print(e)

 
if __name__ == '__main__':
    src_file = 'E:\imageNet\ILSVRC2012_img_train.tar'
    tar_dir = 'E:\imageNet\ILSVRC2012_img_train'
    tar = tarfile.open(src_file, "r")
    tar.extractall(tar_dir)
    
    dst_dir = 'E:\imageNet\ILSVRC2012_img_train' 
    extract(tar_dir, dst_dir)

