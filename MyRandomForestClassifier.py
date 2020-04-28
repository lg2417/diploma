import numpy
from sklearn.ensemble import RandomForestClassifier
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

class MyRandomForestClassifier(RandomForestClassifier):
    
    def __get_hist__(self, img: numpy.ndarray):
        """
        Makes feature vector from a histogram.
        
        :img: 2D numpy array
        
        :return: feature vector
        """
        [x1,y1]=numpy.shape(img)
        img1=numpy.reshape(img,x1*y1)
        hist_1 = plt.hist(img1,bins=[-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8])[0].astype(int)/(x1*y1)
        return(hist_1)
        
    def get_dataset (self,img):
        """
        Makes feature vector by concatenating 9 feature vectors. 
        
        :img: 3D numpy array
        
        :return: feature vector        
        """
        img1 = self.mean_filter(img[:,:,0]).astype(int)
        img2 = self.mean_filter(img[:,:,1]).astype(int)
        img3 = self.mean_filter(img[:,:,2]).astype(int)
       
        original_img=pd.Series(numpy.concatenate([self.__get_hist__(img1),self.__get_hist__(img2),self.__get_hist__(img3)], axis=0))
       
        gaus1 = scipy.ndimage.gaussian_filter(img1,1)
        gaus2 = scipy.ndimage.gaussian_filter(img2,1)
        gaus3 = scipy.ndimage.gaussian_filter(img3,1)
        
        gaus1_img=pd.Series(numpy.concatenate([self.__get_hist__(gaus1),self.__get_hist__(gaus2),self.__get_hist__(gaus3)], axis=0))
        
        gaus4 = scipy.ndimage.gaussian_filter(img1,2)
        gaus5 = scipy.ndimage.gaussian_filter(img2,2)
        gaus6 = scipy.ndimage.gaussian_filter(img3,2)
        
        gaus2_img=pd.Series(numpy.concatenate([self.__get_hist__(gaus4),self.__get_hist__(gaus5),self.__get_hist__(gaus6)], axis=0))
        
        dataset=pd.concat([original_img,gaus1_img,gaus2_img], axis = 0).reset_index().transpose().fillna(0).drop('index')
        return(dataset)
        
    def make_xtest(self,x: list):
        """
        Makes xtest: gets feature vector for every image and concatenates all feature vectors into DataFrame.
        
        :x: list of 3D numpy arrays (photos)
        
        :return: DataFrame of feature vectors
        """
        Xtest=pd.DataFrame()
        for i in range(len(x)):
            img_1=self.get_dataset(x[i])
            Xtest=pd.concat([Xtest,img_1], axis = 0).reset_index().drop(['index'], axis=1)
        return Xtest
    
    def mean_filter (self,img: numpy.ndarray):
        """
        Highlight existing hot pixels
        
        :img: 2D numpy array - photo
        
        :return: 2D numpy array - photo after highlighting hot pixels
        """
        mask = numpy.array([[-1/9,-1/9,-1/9], [-1/9,8/9,-1/9],[-1/9,-1/9,-1/9]], dtype=numpy.float64)
        img_out = signal.convolve2d(img,mask,mode='same').astype(numpy.float64)
        return img_out     

    def predict_proba(self, X):
        """
        Redefined function. Before predicting it's necessary to get feature vectors for each photo.
        
        :x: list of 3D numpy arrays (photos)
        
        :return: predictions
        """
        X1=self.make_xtest(X)
        return super(MyRandomForestClassifier, self).predict_proba(X1)
        
        