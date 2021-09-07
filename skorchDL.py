import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict, Counter

import sys

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as metrics
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score

import joblib

from pprint import pprint

from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier
from skorch import NeuralNetRegressor

class NNClass(nn.Module):
    def __init__(self, nFeat, nHid1=20, nHid2 = 10, drop=0.5, nonlin=F.relu):
        super(NNClass, self).__init__()

        self.dense0 = nn.Linear(nFeat, nHid1)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(drop)
        self.dense1 = nn.Linear(nHid1, nHid2)
        self.dropout = nn.Dropout(drop)
        self.output = nn.Linear(nHid2, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X),dim=1)
        return X

class NNReg(nn.Module):
    def __init__(self, nFeat, nHid1=20, nHid2 = 10, drop=0.5, nonlin=F.relu):
        super(NNReg, self).__init__()

        self.dense0 = nn.Linear(nFeat, nHid1)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(drop)
        self.dense1 = nn.Linear(nHid1, nHid2)
        self.dropout = nn.Dropout(drop)
        self.output = nn.Linear(nHid2, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = self.output(X)
        return X
 
class NNRegDeep(nn.Module):
    
    def __init__(self, nFeat, nHid1=20, nHid2 = 10, drop=0.5, nonlin=F.relu):
        super(NNRegDeep, self).__init__()

        nHid3 = int(np.ceil(0.5*(nHid1+nHid2)))
        self.dense0 = nn.Linear(nFeat, nHid1)
        self.dropout = nn.Dropout(drop)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(nHid1, nHid3)
        self.dense2 = nn.Linear(nHid3, nHid2)
        self.output = nn.Linear(nHid2, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        X = self.output(X)
        
        return X    

class FCNClass():

    def __init__(self,X_train,cls_train,X_test,cls_test):
        
        self.X = X_train
        self.classes = cls_train
        self.X_test = X_test
        self.classes_test = cls_test
        
    def nnClassify(self):
        
        print("\n")
    
    def nnClassifier(self,nH1,nH2,lr,num_epochs):
        
        nn_class = NeuralNetClassifier(NNClass,module__nFeat=self.X.shape[1],module__nHid1=nH1,module__nHid2=nH2,
                               max_epochs=num_epochs,lr=lr,iterator_train__shuffle=True,verbose=0)  
        
        nn_class.fit(self.X.astype(np.float32), self.classes)
        lpred = nn_class.predict(self.X_test.astype(np.float32))
        print("F1 for the NN model : ", f1_score(self.classes_test,lpred))                  
        print("PRECISION for the NN model : ", precision_score(self.classes_test,lpred))          
        print("RECALL for the NN model : ", recall_score(self.classes_test,lpred))          
                
        #scoring = ('f1','precision','recall')
        #nn_class_scores = cross_validate(nn_class,self.X.astype(np.float32), self.classes,scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        
        #print("F1 : ", np.mean(nn_class_scores['test_f1']),"\n",nn_class_scores['test_f1'])
        #print("PRECISION : ", np.mean(nn_class_scores['test_precision']),"\n",nn_class_scores['test_precision'])       
        #print("RECALL : ", np.mean(nn_class_scores['test_recall']),"\n",nn_class_scores['test_recall'])
        
    def optimizeParams(self,description,num_epochs):
        
     
        nn_class = NeuralNetClassifier(NNClass,module__nFeat=self.X.shape[1],module__nHid1=25,
                               max_epochs=num_epochs,lr=0.01,iterator_train__shuffle=True,verbose=0)
        
        params = {'lr': [0.001,0.003,0.006,0.01,0.03,0.06,0.1],
                  'module__nHid1': [30,25,20,15],
                  'module__nHid2': [15,10,5],
                  }
        gs = GridSearchCV(nn_class, params, refit=False, cv=3, scoring='f1',n_jobs=4,verbose=3)

        gs.fit(self.X.astype(np.float32), self.classes)
        print(gs.best_score_, gs.best_params_)
        
        save_file = description+".pkl"
        joblib.dump(gs, save_file)        


class FCNReg():

    def __init__(self,X,y,Xt,yt):
        
        self.X = X
        self.y = y.reshape(-1, 1)
        self.X_test = Xt
        self.ytrue = yt.reshape(-1,1)
           
    def nnRegressor(self,nH1,nH2,lr,num_epochs):
        
        nn_reg = NeuralNetRegressor(NNReg,module__nFeat=self.X.shape[1],module__nHid1=nH1,module__nHid2=nH2,
                               max_epochs=num_epochs,lr=lr,iterator_train__shuffle=True)  
        
        #nn_reg.fit(self.X.astype(np.float32), self.y.astype(np.float32))
        #ypred = nn_reg.predict(self.X_test.astype(np.float32))
        #print("R2 for the NN model : ", r2_score(self.ytrue,self.ypred))                  
        #print("RMS error for the NN model : ", np.sqrt(mean_squared_error(self.ytrue,self.ypred)))          
        
        scoring = ('neg_root_mean_squared_error','r2')
        nn_reg_scores = cross_validate(nn_reg,self.X.astype(np.float32), self.y.astype(np.float32),scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print(nn_reg_scores['test_r2'])
        print("R2 : ", np.mean(nn_reg_scores['test_r2']))
        print("RMSE : ", -np.mean(nn_reg_scores['test_neg_root_mean_squared_error']))        
        
    def optimizeParams(self,description,num_epochs):
        
        nn_reg = NeuralNetRegressor(NNReg,module__nFeat=self.X.shape[1],module__nHid1=25,
                               max_epochs=num_epochs,lr=0.01,iterator_train__shuffle=True,verbose=0)
        
        params = {'lr': [0.001,0.003,0.006,0.01,0.03,0.06,0.1],
                  'module__nHid1': [30,25,20,15],
                  'module__nHid2': [15,10,5],
                  }
        gs = GridSearchCV(nn_reg, params, refit=False, cv=3, scoring='r2',n_jobs=4,verbose=3)

        gs.fit(self.X.astype(np.float32), self.y.astype(np.float32))
        print(gs.best_score_, gs.best_params_)
        
        save_file = description+".pkl"
        joblib.dump(gs, save_file)      
        
        
class FCNClass():

    def __init__(self,X_train,cls_train,X_test,cls_test):
        
        self.X = X_train
        self.classes = cls_train
        self.X_test = X_test
        self.classes_test = cls_test
        
    def nnClassify(self):
        
        print("\n")
    
    def nnClassifier(self,nH1,nH2,lr,num_epochs):
        
        nn_class = NeuralNetClassifier(NNClass,module__nFeat=self.X.shape[1],module__nHid1=nH1,module__nHid2=nH2,
                               max_epochs=num_epochs,lr=lr,iterator_train__shuffle=True)  
        
        nn_class.fit(self.X.astype(np.float32), self.classes)
        lpred = nn_class.predict(self.X_test.astype(np.float32))
        print("F1 for the NN model : ", f1_score(self.classes_test,lpred))                  
        print("PRECISION for the NN model : ", precision_score(self.classes_test,lpred))          
        print("RECALL for the NN model : ", recall_score(self.classes_test,lpred))          
                
        #scoring = ('f1','precision','recall')
        #nn_class_scores = cross_validate(nn_class,self.X.astype(np.float32), self.classes,scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        
        #print("F1 : ", np.mean(nn_class_scores['test_f1']),"\n",nn_class_scores['test_f1'])
        #print("PRECISION : ", np.mean(nn_class_scores['test_precision']),"\n",nn_class_scores['test_precision'])       
        #print("RECALL : ", np.mean(nn_class_scores['test_recall']),"\n",nn_class_scores['test_recall'])
        
    def optimizeParams(self,description,num_epochs):
        
     
        nn_class = NeuralNetClassifier(NNClass,module__nFeat=self.X.shape[1],module__nHid1=25,
                               max_epochs=num_epochs,lr=0.01,iterator_train__shuffle=True,verbose=0)
        
        params = {'lr': [0.001,0.003,0.006,0.01,0.03,0.06,0.1],
                  'module__nHid1': [30,25,20,15],
                  'module__nHid2': [15,10,5],
                  }
        gs = GridSearchCV(nn_class, params, refit=False, cv=3, scoring='f1',n_jobs=4,verbose=3)

        gs.fit(self.X.astype(np.float32), self.classes)
        print(gs.best_score_, gs.best_params_)
        
        save_file = description+".pkl"
        joblib.dump(gs, save_file)        


class FCNRegDeep():

    def __init__(self,X,y,Xt,yt):
        
        self.X = X
        self.y = y.reshape(-1, 1)
        self.X_test = Xt
        self.ytrue = yt.reshape(-1,1)
           
    def nnRegressor(self,nH1,nH2,lr,num_epochs):
        
        nn_reg = NeuralNetRegressor(NNRegDeep,module__nFeat=self.X.shape[1],module__nHid1=nH1,module__nHid2=nH2,
                               max_epochs=num_epochs,lr=lr,iterator_train__shuffle=True)  
        
        nn_reg.fit(self.X.astype(np.float32), self.y.astype(np.float32))
        self.ypred = nn_reg.predict(self.X_test.astype(np.float32))
        print("R2 for the NN model : ", r2_score(self.ytrue,self.ypred))                  
        print("RMS error for the NN model : ", np.sqrt(mean_squared_error(self.ytrue,self.ypred)))          
          
        