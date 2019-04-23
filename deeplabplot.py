#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:46:18 2019

@author: test
"""

from model import Deeplabv3,get_unet
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import os
from keras.utils import plot_model

ISZ=160
N_Cls=6
inDir='/media/test/新加卷/数据/kaggle/'

#打开遥感图像,把轴0放到轴3的位置,channel_last
def M(image_id):
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

#图像拉伸强化
def stretch_n(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands)#out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)#out = np.zeros_like(bands).astype(np.float32)

def predict_id(data, model):

    prd = np.zeros((3200, 3200,N_Cls)).astype(np.float32)
    line = []
    for i in range(0, 39):        
        for j in range(0, 39):
            line.append(data[i * 80:(i+1)*80+80, j * 80:(j+1)*80+80,:])
    x = 2 * np.array(line) - 1  #[-1,1]
    tmp = model.predict(x, batch_size=4)
    k=0
    for i in range(0, 39):        
        for j in range(0, 39):
            
            prd[i * 80:(i+1)*80+80, j * 80:(j+1)*80+80,:] = tmp[k]
            k+=1
    #trs = [0.4,0.4,0.5,0.4,0.4,0.4]
    #trs = [0.4,0.4,0.3,0.6,0.4,0.2]
    #trs = [0.3,0.3,0.4,0.6,0.2,0.2]#x1
    trs = [0.4,0.3,0.3,0.6,0.7,0.2]#x1
    #trs=[0.3,0.2,0.4,0.6,0.6,0.2]#unet
    #trs=[0.4,0.4,0.4,0.2,0.7,0.1]#unet2
    #trs = [0.4,0.4,0.4,0.4,0.5,0.3]
    
    for i in range(N_Cls):
        prd[:,:,i] = prd[:,:,i] > trs[i]
    return prd

def check_predict(id='6120_2_3'):
    model = Deeplabv3(backbone='mobilenetv2')
    #model.load_weights('weights/deeplab_x_jk0.5970')
    msk = predict_id(id, model)
    plt.show()

if __name__ == '__main__':
    data=np.load('test.npy')
    data2=np.load('612022.npy')   
    model = Deeplabv3(backbone='xception')
    model.load_weights('weights/deeplab_x_jk0.6617', by_name=True)
    #model = Deeplabv3(backbone='mobilenetv2')
    #model.load_weights('weights/deeplab_m_jk0.6350')
    msk1 = predict_id(data, model)
    msk2 = predict_id(data2, model)
    tiff.imshow(msk2[:,:,1])
    model2 = get_unet()
    model2.load_weights('weights/unet_jk0.6198')
    msk2 = predict_id(data2, model2)
    for i in range(N_Cls):
        plt.imshow(msk[:,:,1])
        #plt.imsave('picturem/_{}.tif'.format(i),msk[:,:,i])
    #model.summary()
    #plot_model(model,to_file='deeplab_x.png',show_shapes=True)












