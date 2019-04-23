#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
import shapely.wkt
import shapely.affinity

N_Cls = 10
inDir = '/media/test/新加卷2/数据/kaggle/'
DF = pd.read_csv(inDir + 'train_wkt_v4.csv')
GS = pd.read_csv(inDir + 'grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))

ISZ = 160 #输入尺寸
smooth = 1e-12


# project geometry to pixel coordinates
def _convert_coordinates_to_raster(coords, img_size, xymax):     
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int
    
    
#返回0,1行数据
def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)

    
#得到所有的多边形  
def _get_polygon_list(wkt_list_pandas, imageId, cType): 
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList

    
#将多边形转换成轮廓
def _get_and_convert_contours(polygonList, raster_img_size, xymax): 
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list

#在底层中画出轮廓
def _plot_mask_from_contours(raster_img_size, contours, class_value=1): 
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)#(0,0,0)
    return img_mask

#产生底层
def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask

#打开遥感图像并翻转
def M(image_id):
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img
    
def R(image_id):
    filename = os.path.join(inDir, 'three_band', '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img
    
#图像拉伸，归一化到（0，1）
def stretch_n(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = (bands[:, :, i] - c) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)#out = np.zeros_like(bands).astype(np.float32)

#将800*800*8重采样到3200*3200*8，然后和3200*3200*3融合
def nearest(mfile,rgb):
    
    x=np.zeros((3200,3200,8)).astype(np.float32)
    for k in range(8):
        for i in range(800):
            for j in range(800):
                for ii in range(4):
                    x[i*4:i*4+ii+1,j*4:j*4+1+ii,k]=mfile[i,j,k]
    x[0:3200,0:3200,1]=rgb[0:3200,0:3200,2]
    x[0:3200,0:3200,2]=rgb[0:3200,0:3200,1]
    x[0:3200,0:3200,4]=rgb[0:3200,0:3200,0]
    return x.astype(np.float32)

def stick_all_train():
    print ("let's stick all imgs together")
    s = 3200

    x = []    #M.tif has 8 bands
    y = []
    
    ids = sorted(DF.ImageId.unique())    #all images ids
    print (len(ids))    #25
    for i in range(25):
        id = ids[i]
        yy=[]
        img = M(id)
        img = stretch_n(img)
        rgb = R(id)
        rgb = stretch_n(rgb)
        r_size_1=rgb.shape[0]
        r_size_2=rgb.shape[1]
            
        xx=nearest(img,rgb)
            
        x.append(xx)            
        yy.append( generate_mask_for_image_and_class((r_size_1, r_size_2), id, 1)[:s, :s])
        yy.append( generate_mask_for_image_and_class((r_size_1, r_size_2), id, 3)[:s, :s]+
                 generate_mask_for_image_and_class((r_size_1, r_size_2), id, 4)[:s, :s])
        yy.append( generate_mask_for_image_and_class((r_size_1, r_size_2), id, 5)[:s, :s])
        yy.append( generate_mask_for_image_and_class((r_size_1, r_size_2), id, 6)[:s, :s])
        yy.append( generate_mask_for_image_and_class((r_size_1, r_size_2), id, 7)[:s, :s]+
                 generate_mask_for_image_and_class((r_size_1, r_size_2), id, 8)[:s, :s])
        yy.append( generate_mask_for_image_and_class((r_size_1, r_size_2), id, 9)[:s, :s]+
                 generate_mask_for_image_and_class((r_size_1, r_size_2), id, 10)[:s, :s])
        y.append(yy)
    x,y=np.array(x),np.array(y)
    print (x.shape, y.shape)
    y[y>1]=1
    np.save('x_data' , x)
    np.save('y_data' , y)

#筛选，强化以及制作训练数据集
def make_data(P1, P2):
    ISZ = 160
    tr = [0.3, 0.1, 0.25, 0.95, 0.09, 0.001]
    x_train, y_train = [], []
    x_eval, y_eval = [], []
    for k in range(25):
        for i in range(20):
            xc=i*ISZ
            for j in range(20):
                yc=j*ISZ
                im = P1[k,xc:xc + ISZ, yc:yc + ISZ]
                ms = P2[k,xc:xc + ISZ, yc:yc + ISZ]
                for j in range(6):
                    sm = np.sum(ms[:, :, j])
                    if 1.0 * sm / ISZ ** 2 > tr[j]:
                        if random.uniform(0, 1) > 0.5:
                            im = im[::-1]    #axis=0 reverse
                            ms = ms[::-1]
                        if random.uniform(0, 1) > 0.5:
                            im = im[:, ::-1]
                            ms = ms[:, ::-1]    #axis=1 reverse enhance data
                        if random.uniform(0, 1) > 0.2:
                            x_train.append(im)
                            y_train.append(ms)
                        else:
                            x_eval.append(im)
                            y_eval.append(ms)

    x_train, y_train = 2 * np.array(x_train) - 1, np.array(y_train)
    x_eval, y_eval = 2 * np.array(x_eval) - 1, np.array(y_eval)
    print (x_train.shape, y_train.shape,x_eval.shape, y_eval.shape)
    
    np.save('x161_train', x_train)
    np.save('y161_train', y_train)
    np.save('x161_eval', x_eval)
    np.save('y161_eval', y_eval)
    
def make_val():
    print ("let's pick some samples for validation")
    img = np.load('x_data.npy')
    msk = np.load('y_data.npy')
    msk=np.transpose(msk, (0, 2, 3, 1))
    make_data(img,msk)


if __name__ == '__main__':
    make_val()
    make_data()
   
