import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import signal
import pandas as pd
from scipy.ndimage import gaussian_filter

class MyRandomForestClassifier(RandomForestClassifier):
    
    def __get_hist__(self, img: np.ndarray):
        """
        Makes feature vector from a histogram.
        
        :img: 2D numpy array
        
        :return: feature vector
        """
        x1,y1=np.shape(img)
        img1=img.flatten()
        hist = np.histogram(img1,bins=np.arange(-8,9))[0].astype(int)/(x1*y1)
        return(hist)
        
    def get_dataset (self,img):
        """
        Makes feature vector by concatenating 9 feature vectors. 
        
        :img: 3D numpy array
        
        :return: feature vector        
        """
        img1 = self.mean_filter(img[:,:,0]).astype(int)
        img2 = self.mean_filter(img[:,:,1]).astype(int)
        img3 = self.mean_filter(img[:,:,2]).astype(int)
       
        original_img=pd.Series(np.concatenate([self.__get_hist__(img1),self.__get_hist__(img2),self.__get_hist__(img3)], axis=0))
       
        gaus1 = gaussian_filter(img1,1)
        gaus2 = gaussian_filter(img2,1)
        gaus3 = gaussian_filter(img3,1)
        
        gaus1_img=pd.Series(np.concatenate([self.__get_hist__(gaus1),self.__get_hist__(gaus2),self.__get_hist__(gaus3)], axis=0))
        
        gaus4 = gaussian_filter(img1,2)
        gaus5 = gaussian_filter(img2,2)
        gaus6 = gaussian_filter(img3,2)
        
        gaus2_img=pd.Series(np.concatenate([self.__get_hist__(gaus4),self.__get_hist__(gaus5),self.__get_hist__(gaus6)], axis=0))
        
        dataset=pd.concat([original_img,gaus1_img,gaus2_img], axis = 0).reset_index().transpose().fillna(0).drop('index')
        return(dataset)
        
    def make_xtest(self,x: list):
        """
        Makes xtest: gets feature vector for every image and concatenates all feature vectors into DataFrame.
        
        :x: list of 3D numpy arrays (photos)
        
        :return: DataFrame of feature vectors
        """
        Xtest=pd.DataFrame()
        for photo in x:
            img=self.get_dataset(photo)
            Xtest=pd.concat([Xtest,img], axis = 0).reset_index().drop(['index'], axis=1)
        return Xtest
    
    def mean_filter (self,img: np.ndarray):
        """
        Highlight existing hot pixels
        
        :img: 2D numpy array - photo
        
        :return: 2D numpy array - photo after highlighting hot pixels
        """
        mask = np.array([[-1/9,-1/9,-1/9], [-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]], dtype=np.float64)
        img_out = signal.convolve2d(img,mask,mode='same').astype(np.float64)
        return img_out     

    def predict_proba(self, X:list):
        """
        Redefined function. Before predicting it's necessary to get feature vectors for each photo.
        
        :x: list of 3D numpy arrays (photos)
        
        :return: predictions
        """
        Xtest=self.make_xtest(X)
        return super(MyRandomForestClassifier, self).predict_proba(Xtest)
        
        