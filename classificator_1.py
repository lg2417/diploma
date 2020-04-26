from skimage.io import imread, imshow, imsave
import numpy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import scipy
from numba import jit, njit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import feature_importance_permutation

    #Сбор полного Xtrain из сохраненных файлов 
X4=pd.read_pickle("./xtrain_sams2.pkl")
X3=pd.read_pickle("./xtrain_fuji2.pkl")
X2=pd.read_pickle("./xtrain_nikon2.pkl")
X1=pd.read_pickle("./xtrain_huaw2.pkl")
Xtrain=pd.concat([X1,X2,X3,X4], axis = 0).reset_index().drop(['index'], axis=1)
    #Сбор полного Xtest из сохраненных файлов 
#X1=pd.read_pickle("./xtest_sams_dog.pkl")
#X2=pd.read_pickle("./xtest_fuji_dog.pkl")
#X3=pd.read_pickle("./xtest_nikon_dog.pkl")
#X4=pd.read_pickle("./xtest_huaw_dog.pkl")
#Xtest=pd.concat([X1,X2,X3,X4], axis = 0).reset_index().drop(['index'], axis=1) 
Xtest=pd.read_pickle('./xtest_all.pkl')
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

model = RandomForestClassifier(n_estimators =150, min_samples_leaf=3,max_depth=16) 
model.fit(Xtrain, Ytrain)
Y= model.predict(Xtest)
Y1=model.predict_proba(Xtest)

Ytest=[3,4,3,1,4,2,2,1,3,2,4,2,3,1,4,4,3,1,1,2]
#Ytest = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
#         3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
#         4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] 
res1=precision_score(Ytest, Y, average='micro')
res2=recall_score(Ytest, Y, average='micro')
#
from sklearn.metrics import  confusion_matrix
aa=confusion_matrix(Ytest, Y)
importance_vals = model.feature_importances_
print(importance_vals)

std = numpy.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = numpy.argsort(importance_vals)[::-1]
X=numpy.array(Xtest)
# Plot the feature importances of the forest
plt.figure()
plt.title("Random Forest feature importance")
plt.bar(range(X.shape[1]), importance_vals[indices],
        yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, 25])
plt.ylim([0, 0.15])
#plt.show()
plt.savefig('./feat_imp_48.png')
#Ytest =numpy.array ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
#         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
#         3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
#         4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4] )
#
X=numpy.array(Xtest)
imp_vals, imp_all = feature_importance_permutation(
    predict_method=model.predict, 
    X=numpy.array(Xtest),
    y=numpy.array(Ytest),
    metric='accuracy',
    num_rounds=10,
    seed=1)


std = numpy.std(imp_all, axis=1)
indices = numpy.argsort(imp_vals)[::-1]

plt.figure()
plt.title("Random Forest feature importance via permutation importance")
plt.bar(range(X.shape[1]), imp_vals[indices],
        yerr=std[indices])
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, 30])
#plt.show()
plt.savefig('./feat_imp_dog_perm.png')
#aa.to_pickle('./conf_matr.pkl')