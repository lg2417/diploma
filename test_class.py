# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:36:01 2020

@author: Elena
"""

from example import PixelMapClassifier
import pickle

with open('xtest_pixel.data', 'rb') as filehandle:
    # read the data as binary data stream
    xtest_pixel = pickle.load(filehandle)
m1=PixelMapClassifier()
y=m1.predict(xtest_pixel)
#len(xtest_pixel)
#a=xtest_pixel[0]