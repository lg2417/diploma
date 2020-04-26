# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:38:10 2020

@author: Elena
"""

from skimage.io import imread, imshow, imsave
import numpy
import os
from math import floor, ceil
import scipy
from numba import jit
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def cut_img(img,pixel_map):
    [x1,y1]=pixel_map.shape
    [x2,y2]=img.shape
    if (x1<x2)and(y1<y2):
        x3=int((x2-x1)/2)
        y3=int((y2-y1)/2)
        img_p=img[x3:(x3+x1),y3:(y3+y1)]
        pixel_m=pixel_map
    elif (x1>x2)and(y1>y2):
        x3=int((x1-x2)/2)
        y3=int((y1-y2)/2)
        pixel_m=pixel_map[x3:(x3+x2),y3:(y3+y2)]
        img_p=img
    else:
        pixel_m=pixel_map
        img_p=img
    return(img_p,pixel_m)

def mean_filter (img2):
    mask = numpy.array([[-1/9,-1/9,-1/9], [-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]], dtype=numpy.float64)
    img_out = scipy.signal.convolve2d(img2,mask,mode='same')
    return img_out    

@jit
def pirson_coef (pixel_map1, img):

        #в этом блоке сравниваемое изображение или карта пикселей приводятся к одинаковым размерам
        #для определения коэффициента корреляции Пирсона между изображениями используется функция numpy.corrcoef 
    img_p=(img-img.mean())/img.std()
    pixel_map1=(pixel_map1-pixel_map1.mean())/pixel_map1.std()
    [m1,m2]=img_p.shape
    coef_pir=numpy.round(abs(numpy.amax(scipy.signal.correlate(img_p,pixel_map1,mode='same')/m1/m2)),5)
    return(coef_pir)
    
def classificator_on_map (img):
    with open('img_med_huaw.pcl','rb') as fp:
        pixel_map=pickle.load(fp)
    img_p,pixel_map=cut_img(img,pixel_map)
    coef1=pirson_coef(pixel_map,img_p)
    with open('img_med_nikon.pcl','rb') as fp:
        pixel_map=pickle.load(fp)
    img_p,pixel_map=cut_img(img,pixel_map)
    coef2=pirson_coef(pixel_map,img_p)
    with open('img_med_fuji.pcl','rb') as fp:
        pixel_map=pickle.load(fp)
    img_p,pixel_map=cut_img(img,pixel_map)
    coef3=pirson_coef(pixel_map,img_p)
    with open('img_med_sams.pcl','rb') as fp:
        pixel_map=pickle.load(fp)
    img_p,pixel_map=cut_img(img,pixel_map)
    coef4=pirson_coef(pixel_map,img_p)    
    list1=[coef1,coef2,coef3,coef4]
    pred=numpy.argmax(list1)+1
    return list1
    
fds = sorted(os.listdir('./test_all/'))
coefs =[]
for img in fds:
    if img.endswith(('.jpg','.JPG')):
        img_test= mean_filter(imread('./test_all/'+img)[:,:,2]).astype(numpy.float64)
        coef=classificator_on_map(img_test)
        coefs.append(coef)