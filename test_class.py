from sklearn.ensemble import VotingClassifier
from pixelmap import PixelMapClassifier
import pickle
from MyRandomForestClassifier import MyRandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
"""
Stack classifiers RandomForest and PixelMap into classifier with the voiting rule. 
"""
if __name__ == '__main__':
    Xtrain=pd.read_pickle("./Xtrain.pkl")
    with open('xtest_pixel.data', 'rb') as filehandle:
        Xtest = pickle.load(filehandle)
    Ytrain=pd.read_pickle('./Ytain.pkl')
    Name=['PixelMap','RandomForest','Voting','OneVSRest']
    
    cl1=PixelMapClassifier()
    Y_cl1=cl1.predict(Xtest)
    
    cl2=MyRandomForestClassifier(n_estimators =150, min_samples_leaf=3,max_depth=16)
    cl2.fit(Xtrain, Ytrain)
    Y_cl2=cl2.predict(Xtest)
    
    vclf = VotingClassifier(estimators=[ ('PixelMap', cl1), ('RandomForest', cl2)], voting='soft',weights=[1,2])
    vclf.fit(Xtrain,Ytrain)
    Y_vclf=vclf.predict(Xtest)
    
    cl3=OneVsRestClassifier(cl2)
    Y_cl3 = cl3.fit(Xtrain, Ytrain).predict(Xtest)
    
    Ytest=[3,4,3,1,4,2,2,1,3,2,4,2,3,1,4,4,3,1,1,2]
    coef=0
    print('Ytest = ',Ytest,'\n')
    for Y in [Y_cl1,Y_cl2,Y_vclf,Y_cl3]:
        res1=precision_score(Ytest, Y, average='micro')
        res2=recall_score(Ytest, Y, average='micro')
        print('For '+Name[coef]+' classifier:','\n')
        print('precision score = ',res1,'\n')
        print('recall score = ',res2,'\n')
        print('predictions = ',Y,'\n')
        coef+=1