from sklearn.ensemble import VotingClassifier
from example import PixelMapClassifier
import pickle
from MyRandomForestClassifier import MyRandomForestClassifier
import pandas as pd
"""
Stack classifiers RandomForest and PixelMap into classifier with the voiting rule. 
"""
    #Сбор полного Xtrain из сохраненных файлов 
X4=pd.read_pickle("E:/Users/Elena/Desktop/nirs/xtrain_sams.pkl")
X3=pd.read_pickle("E:/Users/Elena/Desktop/nirs/xtrain_fuji.pkl")
X2=pd.read_pickle("E:/Users/Elena/Desktop/nirs/xtrain_nikon.pkl")
X1=pd.read_pickle("E:/Users/Elena/Desktop/nirs/xtrain_huaw.pkl")
Xtrain=pd.concat([X1,X2,X3,X4], axis = 0).reset_index().drop(['index'], axis=1)

with open('xtest_pixel.data', 'rb') as filehandle:
    Xtest = pickle.load(filehandle)
Ytrain = pd.Series()
i=0
n=30*4
for i in range(4*n):
    if i<n:
        Ytrain.loc[i] = 1
        i+=1
    elif i<2*n:
        Ytrain.loc[i] = 2
        i+=1
    elif i<3*n:
        Ytrain.loc[i] = 3
        i+=1
    else: 
        Ytrain.loc[i] = 4
        i+=1

cl1=PixelMapClassifier()
cl2=MyRandomForestClassifier(n_estimators =150, min_samples_leaf=3,max_depth=16)
cl1.fit(Xtrain,Ytrain)
cl2.fit(Xtrain, Ytrain)

eclf = VotingClassifier(estimators=[ ('lr', cl1), ('nb', cl2)], voting='hard')
eclf.fit(Xtrain,Ytrain)
Y=eclf.predict(Xtest)

