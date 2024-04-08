# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:13:22 2022

@author: zhaom
"""
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    PATH = './data/cifar10/cifar-10-batches-py/data_batch_5'
    data = unpickle(PATH)
    x = data[b'data']
    index = 16
    img = x[index].reshape(3,1024)
    red = img[0].reshape(1024,1)
    green = img[1].reshape(1024,1)
    blue = img[2].reshape(1024,1)
    img = np.hstack((red,green,blue))
    img = img.reshape(32,32,3)
    img1 = Image.fromarray(img)
    img1.save('./cifar10_img/'+str(data[b'labels'][index])+'.jpg')
        
