import numpy as np
from pathlib import Path
import pickle
from scipy import signal 

class PixelMapClassifier(object):
    """
        Classifier based on pearson correlation. 
    """
    def __init__(self, pickle_path='C:/1/'):
        self.pickle_path = pickle_path

    def get_params(self, deep=True):
        """
        Necessary function for sickit-learn estimators
        """
        return {"pickle_path": self.pickle_path}
    
    def set_params(self, **parameters):
        """
        Necessary function for sickit-learn estimators
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __cut_img__(self, img: np.ndarray, pixel_map: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Cuts all input data to one size
        
        :param img: 2 dimencional numpy array - photo
        :param pixel_map: 2 dimencional numpy array - map of hot pixels
        
        :return: cropped images
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
        """
        Gets all pixel maps from the path
        
        :return: list of pixel maps
        """
        pixel_maps_paths = list(Path(self.pickle_path).rglob("*.pcl"))
        pixel_maps = []

        for path in pixel_maps_paths:
            with open(path, 'rb') as fp:
                pixel_maps.append(pickle.load(fp))
        return pixel_maps
    
    def pirson_coef(self, pixel_map:np.ndarray,img:np.ndarray)->float:
        """
        Calculates pearson coefficient between photo and pixel map
        
        :pixel_map1: 2D numpy array - map of hot pixels
        :img: 2D numpy array - photo
        
        :return: pearson coefficient
        """
        img_p=(img-img.mean())/img.std()
        pixel_map1=(pixel_map-pixel_map.mean())/pixel_map.std()
        [m1,m2]=img_p.shape
        coef_pir=np.round(abs(np.amax(signal.correlate(img_p,pixel_map1,mode='same')/m1/m2)),5)
        return(coef_pir)
   
    def __calc_coef__(self,img: np.ndarray):
        """
        Calculates pearson coefficients between photo and all pixel maps
        
        :img: 2D numpy array - photo
        
        :return: list of pearson coefficients 
        """
        pixel_maps=self.__get_pixel_maps__()
        coefs=np.array([])
        for pixel_map in pixel_maps:
                img_p,pixel_map=self.__cut_img__(img,pixel_map)
                coef=self.pirson_coef(pixel_map,img_p)
                coefs=np.append(coefs,coef)
        coefs_prob=np.round(coefs*(1/np.sum(coefs)),3) #normalize pearson coefficients 
        return(coefs_prob)

    def mean_filter (self,img: np.ndarray):
        """
        Highlights existing hot pixels
        
        :img: 3D numpy array - photo
        
        :return: 2D numpy array - photo after highlighting hot pixels
        """
        mask = np.array([[-1/9,-1/9,-1/9], [-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]], dtype=np.float64)
        img_out = signal.convolve2d(img[:,:,2],mask,mode='same').astype(np.float64)
        return img_out     
    def predict(self, x: list) -> list:
        """
        Predicts a number of the device that made this photo. It is defined by 
        argument of max probability
        
        :param x: list of 3D numpy arrays - photos - xtest
        
        :return: list of predictions
        """
        predictions=[]
        for i in range(len(x)):
            img=self.mean_filter(x[i])
            coefs=self.__calc_coef__(img)
            pred=np.argmax(coefs)+1
            predictions.append(pred)
        return predictions
    
    def predict_proba(self, x: list) -> list:
        """
        Predicts probabilities by correlation coefficients between photo and all pixel maps. 
        
        :param x: list of 3D numpy arrays - photos - xtest
        
        :return: list of predictions
        """
        predictions=np.empty(4)
        for i in range(len(x)):
            img=self.mean_filter(x[i])
            coefs=self.__calc_coef__(img)
            predictions=np.vstack((predictions,coefs))
        return predictions[1:,:]
    
    def fit(self, X, y, sample_weight=None):
        return self
