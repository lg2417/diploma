import numpy as np
from pathlib import Path
import pickle
#import pandas as pd
#import scipy
from scipy import signal as s1

class PixelMapClassifier(object):
    """
        Some information about class
    """
    def __init__(self, pickle_path='C:/1/'):
        self.pickle_path = pickle_path



    def __cut_img__(self, img: np.ndarray, pixel_map: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Some info about method
        :param img:
        :param pixel_map:
        :return:
        """
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
        

    def __get_pixel_maps__(self):
        pixel_maps_paths = list(Path(self.pickle_path).rglob("*.pcl"))
        pixel_maps = []

        for path in pixel_maps_paths:
            with open(path, 'rb') as fp:
                pixel_maps.append(pickle.load(fp))
        return pixel_maps
    
    def pirson_coef(self, pixel_map1:np.ndarray,img:np.ndarray)->float:
        img_p=(img-img.mean())/img.std()
        pixel_map1=(pixel_map1-pixel_map1.mean())/pixel_map1.std()
        [m1,m2]=img_p.shape
        #coef_pir=np.round(abs(np.amax(scipy.signal.correlate(img_p,pixel_map1,mode='same')/m1/m2)),5)
        coef_pir=np.round(abs(np.amax(s1.correlate(img_p,pixel_map1,mode='same')/m1/m2)),5)
        return(coef_pir)
   
    def __calc_coef__(self,img: np.ndarray):
        pixel_maps=self.__get_pixel_maps__()
        coefs=[]
        for pixel_map in pixel_maps:
                img_p,pixel_map=self.__cut_img__(img,pixel_map)
                coef=self.pirson_coef(pixel_map,img_p)
                coefs.append(coef)
        return(coefs)
        
    def predict(self, x: list) -> list:
        """
        Some info about method
        :param x:
        :return:
        """
        predictions=[]
        for i in range(len(x)):
            img=x[i]
            coefs=self.__calc_coef__(img)
            pred=np.argmax(coefs)+1
            predictions.append(pred)
        return predictions
