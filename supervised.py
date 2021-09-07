import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from time import time
from collections import defaultdict, Counter
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *

import sys

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold

from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor

import sklearn.metrics as metrics
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from mlxtend.evaluate import bias_variance_decomp
from sklearn.inspection import permutation_importance

from pprint import pprint

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier

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
        X = F.softmax(self.output(X))
        return X
    
class Supervised():
    
    
    def __init__(self,X_train,y_train,X_test,y_test,c,logTrans = False, cls_train = None, cls_test = None, thresh = 0.0, ind = None):
        
        self.X = X_train
        self.x_test = X_test
        self.colNames = c
        self.logTransform = logTrans
        self.classes = cls_train
        self.l_test = cls_test
        self.ind = ind
        
        #if(y.shape[1] == 2):
            #print("Throughput included")
            #self.y = y[:,0]
            #self.t = y[:,1]
            
        #else:
        if(self.logTransform == True):
            self.y = np.log(1.00+y_train)
            self.y_test = np.log(1.00+y_test)
        else:
            self.y = y_train
            self.y_test = y_test
            
        self.alpha = 0.2
        self.beta = 0.5
        self.gamma = 0.2
        self.delta = 0.2
        self.thresh = thresh
        
        self.bImbalance = False
        self.bFeatImp = False
    
    def setImbalanceCorrect(self):
        
        self.bImbalance = True
        
    def setFeatImportance(self,fImp):
        
        self.bFeatImp = True
        self.featImp = fImp
        
        
    def correlationAnalysis(self,cor_type = 'Corr', k = 20):
          
        k = min(k,self.X.shape[1])
        print("k-value : ", k)
        imp_order = None
        feat = []
        correl = []
        
        if (cor_type == 'Corr'):
            
            df = pd.DataFrame(self.X)
            df["y"] = self.y
            cor = df.corr()
            corrs_raw = abs(cor["y"])
            corrs_raw = corrs_raw[:-1]
            #colNames = colNames[~np.isnan(corrs_raw)]
            #corrs_raw = corrs_raw[~np.isnan(corrs_raw)]
            self.corrs = np.sort(corrs_raw.values)[::-1]
            #Remove the NaNs
    
            print("Correlation ordering : ")
            imp_order = (np.argsort(corrs_raw.values)[::-1])
            
            for idx in range(k):
                feat.append(self.colNames[imp_order[idx]])
                correl.append(round(self.corrs[idx],3))
                
        elif (cor_type == "MI"):
            mi_raw = mutual_info_regression(self.X,self.y)
            self.corrs = np.sort(mi_raw)[::-1]
            
            print("MI ordering : ")
            imp_order = (np.argsort(mi_raw)[::-1])
            
            for idx in range(k):
                feat.append(self.colNames[imp_order[idx]])
                correl.append(round(self.corrs[idx],3))
     
        #Do feature reduction if needed
        #Xred = None
        #if (k > 0):
            
            #print("Doing filtering based on correlations.")
            #if ( k > self.X.shape[1]):
                #sys.exit("Reduced number of features must be less than total number of features.")
            #else:
                #imp_ord_red = imp_order[:k]
                #Xred = self.X[:,imp_ord_red]
                
        return (feat,correl)
    
    def plotCorrelations(self):
        
        plt.figure()
        plt.plot(np.linspace(1,len(self.corrs),len(self.corrs),dtype=int),self.corrs,color='r',marker = 'o', mS = 4)
        plt.grid()
        plt.title('Correlation (Linear) | Features and Target',fontsize = 25)
        plt.xlabel('Feature Index (sorted)',fontsize=20)
        plt.ylabel('Correlation (Linear) value',fontsize=20)
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)     
        plt.show()
            

    def plot2DSlices(self):
        
        plt.figure()
        ax = plt.gca()
        #nFeat = X.shape[1]
        nFeat = 4
        for idx in range(4,nFeat+4):
            plt.subplot(2,2,idx-3)
            plt.scatter(self.X[:,idx-1],self.y)
            plt.xlabel(self.colNames[idx-1],fontsize = 22)
            plt.ylabel('Target value',fontsize = 22)
            plt.grid()
        
        plt.suptitle("Target vs normalized features.", fontsize = 24)    
        plt.show()        
            
    def linearRegressionsModels(self):

        #y = transform_target(y)
        
        #Split into training validation and test
        x_train, x_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        
        #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)    
    
        #Train the Linear regression models on training set
        reg_ols = LinearRegression().fit(x_train, y_train)
        reg_lasso = Lasso(alpha=0.1).fit(x_train, y_train)
        reg_ridge = Ridge().fit(x_train, y_train)
        
        #Make all 3 predictions
        y_pred_val_ols = reg_ols.predict(x_val)
        y_pred_val_ridge = reg_ridge.predict(x_val)
        y_pred_val_lasso = reg_lasso.predict(x_val)
        
        #MSE for test/validation set
        
        self.ytrue = y_val
        self.ypred = y_pred_val_ridge        
        
        if(self.logTransform == True):
            
            y_val = np.exp(y_val)-1.00
            y_pred_val_ols = np.exp(y_pred_val_ols)-1.00    
            y_pred_val_ridge = np.exp(y_pred_val_ridge)-1.00
            y_pred_val_lasso = np.exp(y_pred_val_lasso)-1.00
            
            self.ytrue = y_val
            self.ypred = y_pred_val_ridge
        
        print("R2 on validation set for OLS : ", r2_score(y_val,y_pred_val_ols))
        print("R2 on validation set for Ridge : ", r2_score(y_val,y_pred_val_ridge))
        print("R2 on validation set for Lasso : ", r2_score(y_val,y_pred_val_lasso))
            
        print("RMS error OLS : ", np.sqrt(mean_squared_error(y_val, y_pred_val_ols)))
        print("RMS error Ridge : ", np.sqrt(mean_squared_error(y_val, y_pred_val_ridge)))
        print("RMS error Lasso : ", np.sqrt(mean_squared_error(y_val, y_pred_val_lasso)))
    
    
    def linModelsCV(self):
        ols = LinearRegression()
        ridge = Ridge(alpha=0.5)
        lasso = Lasso(alpha=0.1)
        
        scoring = ('neg_root_mean_squared_error','r2')
        print("\n")
        print("OLS model for regression")
        ols_scores = cross_validate(ols,self.X, self.y, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print("R2 : ", np.mean(ols_scores['test_r2']))
        print("RMSE : ", -np.mean(ols_scores['test_neg_root_mean_squared_error']))
        
        print("\n")
        print("Ridge model for regression")
        ridge_scores = cross_validate(ridge,self.X, self.y, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print("R2 : ", np.mean(ridge_scores['test_r2']))
        print("RMSE : ", -np.mean(ridge_scores['test_neg_root_mean_squared_error']))
        
        print("\n")
        print("Lasso model for regression")
        lasso_scores = cross_validate(lasso,self.X, self.y, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print("R2 : ", np.mean(lasso_scores['test_r2']))
        print("RMSE : ", -np.mean(lasso_scores['test_neg_root_mean_squared_error']))        
        
    
    def plot_scatter(self):
        
        plt.figure()
        plt.scatter(self.ytrue,self.ypred)
        xy = np.linspace(np.min(self.ytrue),np.max(self.ytrue),100)
        plt.plot(xy,xy, linewidth = 3, linestyle = '--', color = 'k')
        plt.xlabel('Target value (true)',fontsize = 22)
        plt.ylabel('Target value (predicted)',fontsize = 22) 
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)    
        plt.grid()
        plt.show()
    
    def randomForestRegressionModels(self,nTrees,dDepth):
       
        
        print("Number of examples / features overall : ", self.X.shape)

        #Split into training validation and test
        x_train, x_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
        
        #Train the RF regression model on training set
        rf_reg = RandomForestRegressor(n_estimators=nTrees, max_depth=dDepth, random_state=0)
        print("Fitting the RF model")
        rf_reg.fit(x_train, y_train)
        print(rf_reg.feature_importances_)    
        y_pred_val_rf = rf_reg.predict(x_val)
        
        self.ytrue = y_val
        self.ypred = y_pred_val_rf
        
        if (self.logTransform == True):
            y_val = np.exp(y_val)-1.00
            y_pred_val_rf = np.exp(y_pred_val_rf)-1.00
            
            self.ytrue = y_val
            self.ypred = y_pred_val_rf            
       
        print("R2 for random forest model : ", r2_score(self.ytrue,self.ypred))                  
        print("RMS error Random Forest : ", np.sqrt(mean_squared_error(self.ytrue,self.ypred)))    
     
        #self.plot_scatter()
        #print("Doing Cross Validation")
        #scorers = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        #scores = cross_validate(rf_reg, self.X, self.y, cv=5, scoring=scorers)
        #for key,item in scores.items():
            #print(str(key) + ": " + str(-np.mean(item)))
    
    def ETRegressionModels(self,nTrees,dDepth):


        print("Number of examples / features overall : ", self.X.shape)

        #Split into training validation and test
        x_train, x_val, y_train, y_val = train_test_split(self.X, self.y, test_size=self.alpha, random_state=0)

        #Train the RF regression model on training set
        xg_reg = ExtraTreesRegressor(n_estimators=nTrees, max_depth=dDepth, random_state=0)
        xg_reg = self.regression_task(xg_reg, x_train, y_train, x_val, y_val)          
        print("Fitting the XGBoost model")
        xg_reg.fit(x_train, y_train)
        y_pred_val_xg = xg_reg.predict(x_val)

        self.ytrue = y_val
        self.ypred = y_pred_val_xg

        if (self.logTransform == True):
            y_val = np.exp(y_val)-1.00
            y_pred_val_xg = np.exp(y_pred_val_xg)-1.00

            self.ytrue = y_val
            self.ypred = y_pred_val_xg            

        print("R2 for XGBoost model : ", r2_score(self.ytrue,self.ypred))                  
        print("RMS error XGBoost : ", np.sqrt(mean_squared_error(self.ytrue,self.ypred))) 
        
        self.plot_scatter()
    
    
    
    def XGBRegressionModels(self,nTrees,dDepth):
        
        self.X1 = self.X
        
        if (self.bFeatImp == True):
            
            self.X1 = self.X1[:,self.featImp]
            self.x_test = self.x_test[:,self.featImp]

        print("Number of examples / features overall : ", self.X1.shape)

        #Split into training validation and test
        #x_train, x_val, y_train, y_val = train_test_split(self.X1, self.y, test_size=0.25, random_state=0)
        x_train = self.X1
        y_train = self.y
        
        x_val = self.x_test
        y_val = self.y_test

        #Train the RF regression model on training set
        xg_reg = XGBRegressor(n_estimators=nTrees, max_depth=dDepth, random_state=0)
        print("Fitting the XGBoost model")
        xg_reg.fit(x_train, y_train)
        y_pred_val_xg = xg_reg.predict(x_val)

        self.ytrue = y_val
        self.ypred = y_pred_val_xg

        if (self.logTransform == True):
            y_val = np.exp(y_val)-1.00
            y_pred_val_xg = np.exp(y_pred_val_xg)-1.00

            self.ytrue = y_val
            self.ypred = y_pred_val_xg            

        print("R2 for XGBoost model : ", r2_score(self.ytrue,self.ypred))                  
        print("RMS error XGBoost : ", np.sqrt(mean_squared_error(self.ytrue,self.ypred))) 
        feat_imp = []
        
        if (self.bFeatImp == False):
            
            print("\n")
            print("Important features")
            indices = list(range(self.X.shape[1]))
            #perm_imp = permutation_importance(xg_reg, self.X, self.y, n_repeats=10,random_state=0)            
            feat_imp = [x for _,x in sorted(zip(xg_reg.feature_importances_,indices),reverse=True)]
            #feat_imp = [x for _,x in sorted(zip(perm_imp.importances_mean,indices),reverse=True)]
            print(len(feat_imp), " top features")
            print(feat_imp)
            print(self.colNames[list(feat_imp)])
                            
            print("\n") 
            
          
            
            #importances = xg_reg.feature_importances_
            #indices = np.argsort(importances)[::-1][:12]            
            #plt.figure()
            #plt.barh(range(12), importances[indices],color="r", align="center")
            #plt.yticks(range(12), self.colNames[indices])
            #plt.ylim([-1, 12])
            #ax=plt.gca()
            #ax.xaxis.set_tick_params(labelsize=28)
            #ax.yaxis.set_tick_params(labelsize=28)
            #plt.tight_layout()
            #plt.show()            
          
        return feat_imp
        
    def setParamsMultiStage(self,a,b,g,d):
        
        self.alpha = a
        self.beta = b
        self.gamma = g
        self.delta = d
        
       
    def multiStageClassReg(self):
        
        if (self.bFeatImp == True):
            
            self.X1 = self.X
            self.X = self.X1[:,self.featImp]
            print("Selecting only the important features.", self.X.shape[1])
            
        #Create split for train and test
        print("\n")
        print("\n")
        print("Splitting Train/ Test")
        x_train = self.X
        y_train = self.y 
        l_train = self.classes
        
        #Create split for classification and regression
        #Note that in the below split, y_class are not utilized
        #print("Splitting Train into Classification / Regression ")
        x_class, x_reg, l_class, l_reg , y_class, y_reg = train_test_split(x_train, l_train, y_train, test_size=self.beta, random_state=2)
        
        #Now create traning and test sets for classification
        #print("Splitting Classification into CLassification Train / Classification Test")
        x_class_train, x_class_test, l_class_train, l_class_test = train_test_split(x_class, l_class, test_size=self.gamma, random_state=3)
        #print("#Samples for classification training and testing : ",len(l_class_train),len(l_class_test))
        
        print("#Samples for classification training and testing before correction : ",len(l_class_train),len(l_class_test))
        print(sorted(Counter(l_class_train).items()))        

        if (self.bImbalance):
            
            print("Trying to balance the training dataset")
            new_sampler =  SMOTEENN()
            x_train1, l_train1 = new_sampler.fit_sample(x_class_train, l_class_train)
            x_class_train = x_train1
            l_class_train = l_train1
        
        print("#Samples for classification training and testing after correction : ",len(l_class_train),len(l_class_test))
        print(sorted(Counter(l_class_train).items()))        
        
        print("RF model for classification")
        rf_class = RandomForestClassifier(n_estimators=80,max_depth=8,random_state=10)
        rf_class = self.classification_task(rf_class,x_class_train,l_class_train,x_class_test,l_class_test)
        
        print("Extra Trees model for classification")
        et_class = ExtraTreesClassifier(n_estimators=800,max_depth=8, random_state=10)
        et_class = self.classification_task(et_class,x_class_train,l_class_train,x_class_test,l_class_test)

        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=800,max_depth=8,seed=10)
        xg_class = self.classification_task(xg_class,x_class_train,l_class_train,x_class_test,l_class_test)
       
        #Now do the regression training / testing
        #First filter regression data based on classification label
        #print("Total # Samples for Regression (target = 0/1) : ",len(y_reg)) 
        
        #thresh = 0.0
        #if (self.logTransform == True):
            #thresh = 0.75*np.log(1+self.thresh)
        #else:
            #thresh = 0.75*self.thresh
            
        #print("Before threshold adjustment : ", np.sum((y_reg > self.thresh)), " points for regression training.")
                
        #x_reg = x_reg[(y_reg > thresh),:]
        #y_reg = y_reg[(y_reg > thresh)]
        
        #print("After threshold adjustment : ", len(y_reg), " points for regression training.")


        #print("Total # Samples for Regression (target = 1) : ",len(y_reg))   
        #print("Splitting Regression into Regression Train / Regression Test")
        x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg, y_reg, test_size=self.delta, random_state=111)        
        #print("#Samples for Regression training and testing : ",len(y_reg_train),len(y_reg_test))
        
        x_reg_train = x_reg_train[(y_reg_train > self.thresh)]
        y_reg_train = y_reg_train[(y_reg_train > self.thresh)]
        print("Number of training points for regression : ",len(y_reg_train))
        
        print("Random forest model for regression : ")
        rf_reg = RandomForestRegressor(n_estimators=800,max_depth=8, random_state=0)
        rf_reg = self.regression_task(rf_reg, x_reg_train, y_reg_train, x_reg_test, y_reg_test)
        
        #print("Ridge linear model for regression")
        #rd_reg = Ridge(alpha=1.0)
        #rd_reg = self.regression_task(rd_reg,x_reg_train, y_reg_train, x_reg_test, y_reg_test)
        
        print("Extra Trees model for regression")
        et_reg = ExtraTreesRegressor(n_estimators=800,max_depth=8,random_state=0)
        et_reg = self.regression_task(et_reg,x_reg_train, y_reg_train, x_reg_test, y_reg_test)
        
        print("XGBoost model for regression")
        xg_reg = XGBRegressor(n_estimators=800,max_depth=8,random_state=0)
        xg_reg = self.regression_task(xg_reg,x_reg_train, y_reg_train, x_reg_test, y_reg_test)        
        
        #Final testing and error performance
        
        class_models = [rf_class,et_class,xg_class]
        reg_models = [rf_reg, et_reg, xg_reg]
        desc_class = ["RF","ET","XG"]
        desc_reg = ["RF","ET","XG"]
        perf_dict = {}
        feat_imp = set()
        ytrue = None
        ypred = None
        
        for idx1 in range(3):
            for idx2 in range(3):
                
                print("\n")
                title_str = desc_class[idx1] + " for classification and " + desc_reg[idx2] + " for regression"
                print(title_str)
                print("\n")
                #final,f1,r2,rmse = self.combined_task(class_models[idx1], reg_models[idx2], x_test, l_test, y_test)
                ytrue,ypred = self.combined_task(class_models[idx1], reg_models[idx2], self.x_test, self.l_test, self.y_test)
                #model = desc_class[idx1] + "-" + desc_reg[idx2]
                #perf_tuple = tuple((final,f1,r2,rmse))
                #perf_dict[model] = perf_tuple
                

        
        #perf_items = perf_dict.items()
        #perf_items = sorted(perf_items,key=lambda x:x[1][0],reverse=True)
        #print("\n")
        #print("------------------------------------------------------------------------------------------------")
        #pprint(perf_items)       
        #print("------------------------------------------------------------------------------------------------")
        #print("\n")
        #print("------------------------------------------------------------------------------------------------")
        
        #if (self.bFeatImp == False):
            
            #print("\n")
            #print("Important features")
            #for idx in range(2):
                
                #indices = list(range(self.X.shape[1]))
                ##indices = self.colNames
                #top_feat1 = set([x for _,x in sorted(zip(class_models[idx].feature_importances_,indices))][:10])
                #top_feat2 = set([y for _,y in sorted(zip(reg_models[idx].feature_importances_,indices))][:10])
    
                #feat_imp = feat_imp.union(top_feat1.union(top_feat2))
                #print(len(feat_imp), " top features")
                #print(feat_imp)
                #print(self.colNames[list(feat_imp)])
                            
                #print("\n")
    
        #return list(feat_imp)
        return (ytrue,ypred)
    
    
    
    def multiStageClassRegNew(self):
        
        
        x_train = self.X
        y_train = self.y 
        l_train = self.classes
        
        
        print("Number of training samples for classification : ",len(l_train))
        print("RF model for classification")
        rf_class = RandomForestClassifier(n_estimators=500,max_depth=6,random_state=10)
        rf_class = self.classification_task(rf_class,x_train,l_train,self.x_test,self.l_test)
        
        #print("Extra Trees model for classification")
        et_class = ExtraTreesClassifier(n_estimators=500,max_depth=6, random_state=10)
        et_class = self.classification_task(et_class,x_train,l_train,self.x_test,self.l_test)

        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=500,max_depth=6,seed=10)
        xg_class = self.classification_task(xg_class,x_train,l_train,self.x_test,self.l_test)
       
        #Now do the regression training / testing
        
        print("Threshold for regression : ", self.thresh)
        x_reg_train = x_train[(y_train > self.thresh)]
        y_reg_train = y_train[(y_train > self.thresh)]
        
        x_reg_test = self.x_test[(self.y_test > self.thresh)]
        y_reg_test = self.y_test[(self.y_test > self.thresh)]
        
        print("Number of training points for regression : ",len(y_reg_train))
        
        print("Random forest model for regression : ")
        rf_reg = RandomForestRegressor(n_estimators=500,max_depth=6, random_state=0)
        rf_reg = self.regression_task(rf_reg, x_reg_train, y_reg_train, x_reg_test, y_reg_test)

        #print("Extra Trees model for regression")
        et_reg = ExtraTreesRegressor(n_estimators=500,max_depth=6,random_state=0)
        et_reg = self.regression_task(et_reg,x_reg_train, y_reg_train, x_reg_test, y_reg_test)
        
        print("XGBoost model for regression")
        xg_reg = XGBRegressor(n_estimators=500,max_depth=6,random_state=0)
        xg_reg = self.regression_task(xg_reg,x_reg_train, y_reg_train, x_reg_test, y_reg_test)        
        
        #Final testing and error performance
        
        #class_models = [rf_class,et_class,xg_class]
        #reg_models = [rf_reg, et_reg, xg_reg]
        #desc_class = ["RF","ET","XG"]
        #desc_reg = ["RF","ET","XG"]
        
        class_models = list([rf_class,et_class,xg_class])
        reg_models = list([rf_reg, et_reg, xg_reg])
        desc_class = list(["RF","ET","XG"])
        desc_reg = list(["RF","ET","XG"])
        
        for idx1 in range(3):
            for idx2 in range(3):
                
                print("\n")
                title_str = desc_class[idx1] + " for classification and " + desc_reg[idx2] + " for regression"
                print(title_str)
                ytrue,ypred = self.combined_task(class_models[idx1], reg_models[idx2], self.x_test, self.l_test, self.y_test)
   
    
    def multiStageClassRegNewEval(self, nTrees = 500, dDepth = 6, class_idx = None, reg_idx = None):
        
        
        x_train = self.X
        y_train = self.y 
        l_train = self.classes
        
        
        print("Number of training samples for classification : ",len(l_train))
        print("RF model for classification")
        rf_class = RandomForestClassifier(n_estimators=nTrees,max_depth=dDepth,random_state=10)
        rf_class = self.classification_task(rf_class,x_train,l_train,self.x_test,self.l_test)
        
        #print("Extra Trees model for classification")
        et_class = ExtraTreesClassifier(n_estimators=nTrees,max_depth=dDepth, random_state=10)
        #et_class = self.classification_task(et_class,x_train,l_train,self.x_test,self.l_test)

        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=nTrees,max_depth=dDepth,seed=10)
        xg_class = self.classification_task(xg_class,x_train,l_train,self.x_test,self.l_test)
       
        #Now do the regression training / testing
        
        print("Threshold for regression : ", self.thresh)
        x_reg_train = x_train[(y_train > self.thresh)]
        y_reg_train = y_train[(y_train > self.thresh)]
        
        x_reg_test = self.x_test[(self.y_test > self.thresh)]
        y_reg_test = self.y_test[(self.y_test > self.thresh)]
        
        print("Number of training points for regression : ",len(y_reg_train))
        
        print("Random forest model for regression : ")
        rf_reg = RandomForestRegressor(n_estimators=nTrees,max_depth=dDepth, random_state=0)
        rf_reg = self.regression_task(rf_reg, x_reg_train, y_reg_train, x_reg_test, y_reg_test)

        #print("Extra Trees model for regression")
        et_reg = ExtraTreesRegressor(n_estimators=nTrees,max_depth=dDepth,random_state=0)
        #et_reg = self.regression_task(et_reg,x_reg_train, y_reg_train, x_reg_test, y_reg_test)
        
        print("XGBoost model for regression")
        xg_reg = XGBRegressor(n_estimators=nTrees,max_depth=dDepth,random_state=0)
        xg_reg = self.regression_task(xg_reg,x_reg_train, y_reg_train, x_reg_test, y_reg_test)        
        
        #Final testing and error performance
        
        #class_models = [rf_class,et_class,xg_class]
        #reg_models = [rf_reg, et_reg, xg_reg]
        #desc_class = ["RF","ET","XG"]
        #desc_reg = ["RF","ET","XG"]
        
        class_models = list([rf_class,et_class,xg_class])
        reg_models = list([rf_reg, et_reg, xg_reg])
        desc_class = list(["RF","ET","XG"])
        desc_reg = list(["RF","ET","XG"])
        
        if ((class_idx is not None) and (reg_idx is not None)):
            
            class_models = class_models[class_idx]
            desc_class = desc_class[class_idx]
        
            reg_models = reg_models[reg_idx]
            desc_reg = desc_reg[reg_idx]        
        
        perf_dict = {}
        feat_imp = set()
        ytrue = None
        ypred = None
        ytopdown = None
        
   
        print("\n")
        title_str = desc_class + " for classification and " + desc_reg + " for regression"
        print(title_str)
        print("\n")
        ytrue,ypred,ytopdown = self.combined_task(class_models, reg_models, self.x_test, self.l_test, self.y_test)

        return (ytrue,ypred,ytopdown)
    
    def XGBRegressionEval(self,nTrees,dDepth):

        
        #Train the RF regression model on training set
        xg_reg = XGBRegressor(n_estimators=nTrees, max_depth=dDepth, random_state=10)
        print("Fitting the XGBoost model")
        xg_reg.fit(self.X,self.y)
        #xt = self.x_test[(self.y_test > self.thresh)]
        #yt = self.y_test[(self.y_test > self.thresh)]
        
        xt = self.x_test
        yt = self.y_test
        
        ypred = xg_reg.predict(xt)
        
        if (self.logTransform == True):
            yt = np.exp(yt)-1.00
            ypred = np.exp(ypred)-1.00        


        print("R2 for XGBoost model : ", r2_score(yt,ypred))                  
        print("RMS error XGBoost : ", np.sqrt(mean_squared_error(yt,ypred))) 
                   
        return (yt,ypred)    
    
    def classification_task(self, model,x_train,l_train,x_test,l_test):
        

        
        #print("Model training.")
        model.fit(x_train,l_train)
        l_pred = model.predict(x_test)
        conf_mat = confusion_matrix(l_test,l_pred)
        print("Confusion matrix for the current model : ")
        print(conf_mat)
        print("Classification accuracy : ", accuracy_score(l_test,l_pred)) 
        print("Precision score = ",precision_score(l_test,l_pred))
        print("Recall score = ",recall_score(l_test,l_pred))
        print("F1 score = ",f1_score(l_test,l_pred))
        
        return model
        
    
    
    def regression_task(self, model,x_train,y_train,x_test,y_test):
        
        print("Model training.")
        model.fit(x_train,y_train)
        #print(model.feature_importances_)    
        print("R2 for the model : ", model.score(x_test,y_test))
        y_pred = model.predict(x_test)
        
        if (self.logTransform == True):
            y_test = np.exp(y_test)-1.00
            y_pred = np.exp(y_pred)-1.00
                
        print("RMS error for the model : ", np.sqrt(mean_squared_error(y_test,y_pred)))    
        print('RMS error / Mean(target) ration : ', np.sqrt(mean_squared_error(y_test,y_pred))/np.std(self.y))
        return model    
    
    def combined_task(self,c_model,r_model,x_test,l_test,y_test):
        
        print("Total # of test points : ",len(l_test))
        start = time()
        l_pred = c_model.predict(x_test)
        end = time()
        print("Classification time per instance : ", (end-start)/len(l_test))
        
        
        conf_mat = confusion_matrix(l_test,l_pred)
        print("Confusion matrix for the current model : ")
        print(conf_mat)
        
        print("Final Classification accuracy for the current model : ", accuracy_score(l_test,l_pred))    
        print("Final Precision score = ",precision_score(l_test,l_pred))
        print("Final Recall score = ",recall_score(l_test,l_pred))
        print("Final F1 score = ",f1_score(l_test,l_pred))
        
       
        #Get those with label = 1 
        #x_test_l1 = x_test[((l_pred == 1) & (l_test ==1)),:]
        #y_test_l1 = y_test[((l_pred == 1) & (l_test ==1))]
        x_test_l1 = x_test[((l_pred == 1)),:]
        ytrue = y_test[((l_pred == 1))]
        y_topdown = self.ind[((l_pred == 1))]


        print("Total # of test points : ",len(ytrue))
        start = time()        
        ypred = r_model.predict(x_test_l1)
        end = time()
        print("Regression time per instance : ", (end-start)/len(ytrue))

        self.errorAnalysis(ytrue, ypred, x_test_l1)

        #plt.figure()
        #plt.hist(y_pred_l1[(y_pred_l1<3.0)],bins=50)
        #plt.show()
        
        #y_test_true = y_test[(l_test ==1)]
        #y_test_both = y_test[((l_pred == 1) & (l_test ==1))]
        #print("# Testing points  for the final phase (True, Pred, Both): ",len(y_test_true),len(y_test_l1),len(y_test_both))
        
        if (self.logTransform == True):
            ytrue = np.exp(ytrue)-1.00
            ypred = np.exp(ypred)-1.00        
        
        print("ML prediction : ")
        print("Final R2 for the model : ", r2_score(ytrue,ypred))
        print("Rel. RMS error for the model : ", np.sqrt(mean_squared_error(ytrue,ypred))/np.std(ytrue))
        print("MAPE : ", mean_absolute_percentage_error(ytrue,ypred))
        print("Abs. med. error : ", median_absolute_error(ytrue,ypred))
        print("Rel. med. error (STD) : ", median_absolute_error(ytrue,ypred)/np.std(ytrue))
        rel_errors = np.abs(ytrue-ypred)/np.abs(0.0001+ytrue)
        print("Median rel. error = ",np.median(rel_errors))
        
        print("Top-down prediction : ")
        print("Final R2 for the model : ", r2_score(ytrue,y_topdown))
        print("Rel. RMS error for the model : ", np.sqrt(mean_squared_error(ytrue,y_topdown))/np.std(ytrue))
        print("MAPE : ", mean_absolute_percentage_error(ytrue,y_topdown))
        print("Abs. med. error : ", median_absolute_error(ytrue,y_topdown))
        print("Rel. med. error (STD) : ", median_absolute_error(ytrue,y_topdown)/np.std(ytrue))
        rel_errors = np.abs(ytrue-y_topdown)/np.abs(0.0001+ytrue)
        print("Median rel. error = ",np.median(rel_errors))      
        
                
        return (ytrue,ypred,y_topdown)

        
    def classification_allModels(self):
        
        #Create split for train
        print("Splitting Train/ Test")
        x_train1, x_test, l_train1, l_test = train_test_split(self.X, self.classes, test_size=self.alpha, random_state=1)
        print("#Samples for classification training and testing before correction : ",len(l_train1),len(l_test))
        
        #print(sorted(Counter(l_train1).items()))
        #new_sampler =  KMeansSMOTE()
        #x_train, l_train = new_sampler.fit_sample(x_train1, l_train1)
        x_train = x_train1
        l_train = l_train1
        
        #print("#Samples for classification training and testing after correction : ",len(l_train),len(l_test))
        #print(sorted(Counter(l_train).items()))

        #start = time()
        #print("\n")
        #print("RF model for classification")
        #rf_class = RandomForestClassifier(n_estimators=500,random_state=0)
        #rf_class = self.classification_task(rf_class,x_train,l_train,x_test,l_test)
        ##print(rf_class.feature_importances_)
        #print("Time taken = ", time()-start)
        
        #start = time()
        #print("\n")
        #print("Extra Trees model for classification")
        #et_class = ExtraTreesClassifier(n_estimators=500, random_state=0)
        #et_class = self.classification_task(et_class,x_train,l_train,x_test,l_test)
        ##print(et_class.feature_importances_)
        #print("Time taken = ", time()-start)

        #start = time()
        #print("\n")
        #print("XGBoost model for classification") 
        #xg_class = XGBClassifier(n_estimators=500,seed=0)
        #xg_class = self.classification_task(xg_class,x_train,l_train,x_test,l_test)
        ##print(xg_class.feature_importances_)
        #print("Time taken = ", time()-start)
        
        start = time()
        print("\n")
        print("NN model for classification") 
        nn_class = NeuralNetClassifier(NNClass,module__nFeat=x_train.shape[1],module__nHid1=25,
                                       max_epochs=200,lr=0.043,iterator_train__shuffle=True,)
        nn_class = self.classification_task(nn_class,x_train.astype(np.float32),l_train,x_test.astype(np.float32),l_test)
        print("Time taken = ", time()-start)
        
        
    def regression_allRFModels(self):
        
        #Create split for train
        print("Splitting Train/ Test")
        x_train1, x_test, y_train1, y_test = train_test_split(self.X, self.y, test_size=self.alpha, random_state=1)
        #print("#Samples for classification training and testing before correction : ",len(l_train1),len(l_test))
        
        #print(sorted(Counter(l_train1).items()))
        #new_sampler =  KMeansSMOTE()
        #x_train, l_train = new_sampler.fit_sample(x_train1, l_train1)
        x_train = x_train1
        y_train = y_train1
        
        #print("#Samples for classification training and testing after correction : ",len(l_train),len(l_test))
        #print(sorted(Counter(l_train).items()))

        print("\n")
        print("RF model for regression")
        rf_reg = RandomForestRegressor(n_estimators=600,max_depth=6,random_state=100)
        rf_reg = self.regression_task(rf_reg,x_train,y_train,x_test,y_test)
        #print(rf_reg.feature_importances_)
        
        print("\n")
        print("Extra Trees model for regression")
        et_reg = ExtraTreesRegressor(n_estimators=600,max_depth=6, random_state=100)
        et_reg = self.regression_task(et_reg,x_train,y_train,x_test,y_test)
        #print(et_reg.feature_importances_)

        print("\n")
        print("XGBoost model for regression") 
        xg_reg = XGBRegressor(n_estimators=600,max_depth=6,seed=100)
        xg_reg = self.regression_task(xg_reg,x_train,y_train,x_test,y_test)
        #print(xg_reg.feature_importances_)
    
   
    def rf_models_CV_Reg(self):
        
        scoring = ('neg_root_mean_squared_error','r2')
        print("\n")
        print("RF model for regression")
        rf_reg = RandomForestRegressor(n_estimators=600,max_depth=6,random_state=100)
        rf_reg_scores = cross_validate(rf_reg,self.X, self.y, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print(rf_reg_scores['test_r2'])
        print("R2 : ", np.mean(rf_reg_scores['test_r2']))
        print("RMSE : ", -np.mean(rf_reg_scores['test_neg_root_mean_squared_error']))
        
        print("\n")
        print("Extra Trees model for regression")
        et_reg = ExtraTreesRegressor(n_estimators=600,max_depth=6, random_state=100)
        et_reg_scores = cross_validate(et_reg,self.X, self.y, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print(et_reg_scores['test_r2'])
        print("R2 : ", np.mean(et_reg_scores['test_r2']))
        print("RMSE : ", -np.mean(et_reg_scores['test_neg_root_mean_squared_error']))        

        print("\n")
        print("XGBoost model for regression") 
        xg_reg = XGBRegressor(n_estimators=600,max_depth=6,seed=100)
        xg_reg_scores = cross_validate(xg_reg,self.X, self.y, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print(xg_reg_scores['test_r2'])
        print("R2 : ", np.mean(xg_reg_scores['test_r2']))
        print("RMSE : ", -np.mean(xg_reg_scores['test_neg_root_mean_squared_error']))        
        

    def rf_models_CV_class(self):
        
        scoring = ('f1','precision','recall')
        print("\n")
        
        print("LR model for classification")
        lr_class = SGDClassifier(loss ='log', max_iter=1000, tol=1e-3)
        lr_class_scores = cross_validate(lr_class,self.X, self.classes, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print("F1 : ", np.mean(lr_class_scores['test_f1']))
        print("PRECISION : ", np.mean(lr_class_scores['test_precision']))
        print("RECALL : ", np.mean(lr_class_scores['test_recall']))
        
        print("RF model for classification")
        rf_class = RandomForestClassifier(n_estimators=500,max_depth=8,random_state=100)
        rf_class_scores = cross_validate(rf_class,self.X, self.classes, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print("F1 : ", np.mean(rf_class_scores['test_f1']))
        print("PRECISION : ", np.mean(rf_class_scores['test_precision']))
        print("RECALL : ", np.mean(rf_class_scores['test_recall']))
        
        print("\n")
        print("Extra Trees model for classification")
        et_class = ExtraTreesClassifier(n_estimators=500,max_depth=8, random_state=100)
        et_class_scores = cross_validate(et_class,self.X, self.classes, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print("F1 : ", np.mean(et_class_scores['test_f1']))
        print("PRECISION : ", np.mean(et_class_scores['test_precision']))
        print("RECALL : ", np.mean(et_class_scores['test_recall']))
        
        print("\n")
        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=500,max_depth=8,seed=100)
        xg_class_scores = cross_validate(xg_class,self.X, self.classes, scoring=scoring, cv=KFold(n_splits=5,shuffle=True))
        print("F1 : ", np.mean(xg_class_scores['test_f1']))
        print("PRECISION : ", np.mean(xg_class_scores['test_precision']))
        print("RECALL : ", np.mean(xg_class_scores['test_recall'])) 
        
    def get_fp_fn(self):
        
        indices_full = range(len(self.classes))
        x_train, x_test, l_train, l_test , y_train, y_test,ind_train,ind_test = train_test_split(self.X, self.classes, self.y, indices_full, test_size=self.alpha, random_state=1)

        model = XGBClassifier(n_estimators=1000,seed=0)
        #model = RandomForestClassifier(n_estimators=1000,random_state=0)
        print("Model training.")
        model.fit(x_train,l_train)
        pred_labels = model.predict(x_test)
        top_feat = np.flip(np.argsort(model.feature_importances_))[:10]
        top_col_names = self.colNames[top_feat]
        print(top_col_names)
        print(model.feature_importances_[top_feat])
        
        fp_inds = []
        fn_inds = []
        y_vals_fp = []
        y_vals_fn = []
        
        for idx in range(len(l_test)):
            
            if ((l_test[idx] == 0 ) and (pred_labels[idx] == 1)):
                fp_inds.append(ind_test[idx])
                y_vals_fp.append(y_test[idx])
                
            if ((l_test[idx] == 1 ) and (pred_labels[idx] == 0)):
                fn_inds.append(ind_test[idx])
                y_vals_fn.append(y_test[idx])
                
        print("Number of false positives : ", len(fp_inds))
        print("Number of false negatives : ", len(fn_inds))              
            
        conf_mat = confusion_matrix(l_test,pred_labels)
        print("Confusion matrix for the current model : ")
        print(conf_mat)        
    
        return (fp_inds,fn_inds,y_vals_fp,y_vals_fn,top_col_names)    
    
    
    def roc_auc(self):
        
        #Create split for train
        print("Splitting Train/ Test")
        x_train, x_test, l_train, l_test , y_train, y_test = train_test_split(self.X, self.classes, self.y, test_size=self.alpha, random_state=1)
               
        print("RF model for classification")
        rf_class = RandomForestClassifier(n_estimators=1000,random_state=0)
        rf_class.fit(x_train,l_train)
        metrics.plot_roc_curve(rf_class, x_test, l_test)
        plt.show()                                        
        
       
        print("Extra Trees model for classification")
        et_class = ExtraTreesClassifier(n_estimators=1000, random_state=0)
        et_class.fit(x_train,l_train)
        metrics.plot_roc_curve(et_class, x_test, l_test)
        plt.show()         
        
        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=1000,seed=0)
        xg_class.fit(x_train,l_train)
        metrics.plot_roc_curve(xg_class, x_test, l_test)
        plt.show()         
        
    def prec_recall(self):
        
        #Create split for train
        print("Splitting Train/ Test")
        x_train, x_test, l_train, l_test , y_train, y_test = train_test_split(self.X, self.classes, self.y, test_size=self.alpha, random_state=1)
               
        print("RF model for classification")
        rf_class = RandomForestClassifier(n_estimators=1000,random_state=0)
        rf_class.fit(x_train,l_train)
        metrics.plot_precision_recall_curve(rf_class, x_test, l_test)
        plt.show() 
        
       
        print("Extra Trees model for classification")
        et_class = ExtraTreesClassifier(n_estimators=1000, random_state=0)
        et_class.fit(x_train,l_train)
        metrics.plot_precision_recall_curve(et_class, x_test, l_test)
        plt.show() 
        
        
        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=1000,seed=0)
        xg_class.fit(x_train,l_train)
        metrics.plot_precision_recall_curve(xg_class, x_test, l_test)
        plt.show() 
        
    def bias_variance_decomp_regression(self):
        
        x_tr, x_tst, y_tr, y_tst = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
        
        print("Random Forest Regressor")
        rf_reg = RandomForestRegressor(n_estimators=100,max_depth=6)
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(rf_reg,x_tr,y_tr,x_tst,y_tst,loss='mse',random_seed=1)
        print('Average expected loss: %.3f' % avg_expected_loss)
        print('Average bias: %.3f' % avg_bias)
        print('Average variance: %.3f' % avg_var)  
        print("\n\n")
        

        print("Extra Trees Regressor")
        et_reg = ExtraTreesRegressor(n_estimators=100,max_depth=6)
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(et_reg,x_tr,y_tr,x_tst,y_tst,loss='mse',random_seed=1)
        print('Average expected loss: %.3f' % avg_expected_loss)
        print('Average bias: %.3f' % avg_bias)
        print('Average variance: %.3f' % avg_var)  
        print("\n\n")
        

        print("XGBoost Regressor")
        xg_reg = XGBRegressor(n_estimators=100,max_depth=6)
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(xg_reg,x_tr,y_tr,x_tst,y_tst,loss='mse',random_seed=1)
        print('Average expected loss: %.3f' % avg_expected_loss)
        print('Average bias: %.3f' % avg_bias)
        print('Average variance: %.3f' % avg_var)  
        print("\n\n")        
        
        
    def run_bias_variance_tradeoff_regression(self):
        
        x_tr, x_tst, y_tr, y_tst = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        
        n_est = np.linspace(100,1000,num=19,endpoint=True,dtype=int)
        bias = np.zeros(len(n_est))
        var = np.zeros(len(n_est))
        loss = np.zeros(len(n_est))
        
        start = time()
        
        for idx in range(len(n_est)):
            
            print("Iteration : ",idx)
            print("Time = ", time()-start)
            reg = RandomForestRegressor(n_estimators=n_est[idx],max_depth=8)
            x_tr1, x_tst1, y_tr1, y_tst1 = train_test_split(x_tr, y_tr, test_size=0.5, random_state=idx+1)
            loss1, bias1, var1 = bias_variance_decomp(reg,x_tr1,y_tr1,x_tst,y_tst,loss='mse',random_seed=idx+11)
            bias[idx] = bias1
            var[idx] = var1
            loss[idx] = loss1
        
        print("Time taken for all runs = ", time()-start)    
        np.savez("bv_rf.npz",n_est=n_est,loss=loss,bias=bias,var=var)    
        print("Done saving arrays.")
        
        
        
    def multiStageClassRegThrpt(self):
        
        print("\n")
        print("\n")
        print("Splitting Train/ Test")
        x_train, x_test, l_train, l_test , y_train, y_test, t_train, t_test = train_test_split(self.X, self.classes, self.y, self.t, test_size=self.alpha, random_state=1)
    
        x_class, x_reg, l_class, l_reg , y_class, y_reg = train_test_split(x_train, l_train, y_train, test_size=self.beta, random_state=2)
        x_class_train, x_class_test, l_class_train, l_class_test = train_test_split(x_class, l_class, test_size=self.gamma, random_state=3)
      
        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=1000,max_depth=8,seed=10)
        xg_class = self.classification_task(xg_class,x_class_train,l_class_train,x_class_test,l_class_test)
       
        thresh = 0.0
        if (self.logTransform == True):
            thresh = 0.75*np.log(1+self.thresh)
        else:
            thresh = 0.75*self.thresh
            
        x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg, y_reg, test_size=self.delta, random_state=111)        
        
        print("Random forest model for regression : ")
        rf_reg = RandomForestRegressor(n_estimators=1000,max_depth=8, random_state=0)
        rf_reg = self.regression_task(rf_reg, x_reg_train, y_reg_train, x_reg_test, y_reg_test)
        
        ytrue,ypred = self.combined_task_thrpt(xg_class, rf_reg, x_test, l_test, y_test, t_test)
        
                
        return (ytrue,ypred) 
    
    
    def combined_task_thrpt(self,c_model,r_model,x_test,l_test,y_test, t_test):
        
        l_pred = c_model.predict(x_test)
        conf_mat = confusion_matrix(l_test,l_pred)
        print("Confusion matrix for the current model : ")
        print(conf_mat)
        
        print("Final Classification accuracy for the current model : ", accuracy_score(l_test,l_pred))    
        print("Final Precision score = ",precision_score(l_test,l_pred))
        print("Final Recall score = ",recall_score(l_test,l_pred))
        print("Final F1 score = ",f1_score(l_test,l_pred))
        
       
        #Get those with label = 1 
        #x_test_l1 = x_test[((l_pred == 1) & (l_test ==1)),:]
        #y_test_l1 = y_test[((l_pred == 1) & (l_test ==1))]
        x_test_l1 = x_test[((l_pred == 1)),:]
        y_test_l1 = y_test[((l_pred == 1))]
        t_test_l1 = t_test[((l_pred == 1))]
        
        y_pred_l1 = r_model.predict(x_test_l1)
        
        #plt.figure()
        #plt.hist(y_pred_l1[(y_pred_l1<3.0)],bins=50)
        #plt.show()
        
        print("# Testing points  for the final phase : ",len(y_test),len(y_test_l1))
        
        if (self.logTransform == True):
            y_test_l1 = np.exp(y_test_l1)-1.00
            y_pred_l1 = np.exp(y_pred_l1)-1.00        
        
        print("Final R2 for the model : ", r2_score(y_test_l1,y_pred_l1))
        
           
        #print("RMS error for the model : ", np.sqrt(mean_squared_error(y_test_l1,y_pred_l1))) 
        
        #final_score = 0.5*f1_score(l_test,l_pred) + 0.5*r2_score(y_test_l1,y_pred_l1)
        #print("Final score = ",final_score)
        #self.ytrue = y_test_l1
        #self.ypred = y_pred_l1
        #self.plot_scatter()
        
        #return (final_score,f1_score(l_test,l_pred),r2_score(y_test_l1,y_pred_l1),np.sqrt(mean_squared_error(y_test_l1,y_pred_l1)))
        return (y_test_l1*t_test_l1,y_pred_l1*t_test_l1)    
    
        
    def XGBRegressionModelsEvalThrpt(self,nTrees,dDepth):

        
        #Split into training validation and test
        x_train, x_val, y_train, y_val, t_train, t_val = train_test_split(self.X, self.y, self.t,test_size=0.25, random_state=0)

        #Train the RF regression model on training set
        xg_reg = XGBRegressor(n_estimators=nTrees, max_depth=dDepth, random_state=0)
        print("Fitting the XGBoost model")
        xg_reg.fit(x_train, y_train)
        y_pred_val_xg = xg_reg.predict(x_val)

        y_val = y_val[(y_pred_val_xg > 0.61)]
        t_val = t_val[(y_pred_val_xg > 0.61)]
        y_pred_val_xg = y_pred_val_xg[(y_pred_val_xg > 0.61)]
        
        self.ytrue = y_val
        self.ypred = y_pred_val_xg

        print("R2 for XGBoost model : ", r2_score(self.ytrue,self.ypred))                  
        print("RMS error XGBoost : ", np.sqrt(mean_squared_error(self.ytrue,self.ypred))) 
                   

        return (self.ytrue*t_val,self.ypred*t_val)


    def errorAnalysis(self,yt, yp, x):

        #Considering the large CPI case

        yp = yp[(yt > 1.0)]
        yt = yt[(yt > 1.0)]

        print("\n")
        print("Error analysis : ")
        print("Number of data points : ", len(yt), len(yp))

        res = (yp-yt)
        err = np.abs(res/(0.000001+yt))
        plt.hist(err[err<1.99],bins=100)
        plt.xlabel("Relative error ", fontsize=20)
        plt.ylabel("Counts", fontsize=20)
        plt.show()

        print("Errors for various percentiles")
        percentiles = [10,20,30,40,50,60,70,80,90,100]
        cutoff_err = []
        for per in percentiles:
            cutoff = np.percentile(err,per)
            _err = np.median(err[err<cutoff])
            cutoff_err.append(_err)

        plt.figure()
        plt.plot(percentiles,cutoff_err,'*')
        plt.xlabel("Percentile w.r.t the relative error", fontsize=20)
        plt.ylabel("Median relative error ", fontsize=20)
        plt.show()

        # df = pd.DataFrame(x)
        # df["y"] = err
        #
        # print("Correlations for large errors")
        # #df = df[df["y"] > 1.00]
        # #print(df.shape)
        #
        # cor = df.corr()
        # corrs_raw = abs(cor["y"])
        # corrs_raw = corrs_raw[:-1]
        # # colNames = colNames[~np.isnan(corrs_raw)]
        # # corrs_raw = corrs_raw[~np.isnan(corrs_raw)]
        # corrs = np.sort(corrs_raw.values)[::-1]
        # # Remove the NaNs
        #
        # print("Correlation ordering : ")
        # imp_order = (np.argsort(corrs_raw.values)[::-1])
        # feat = []
        # correl = []
        # k=15
        #
        # for idx in range(k):
        #     feat.append(self.colNames[imp_order[idx]])
        #     correl.append(round(corrs[idx], 3))
        #
        # for idx in range(k):
        #     feat.append(self.colNames[imp_order[idx]])
        #     correl.append(round(corrs[idx], 3))
        #
        # feat_correl = zip(feat,correl)
        #
        # print("\n")
        # print(list(feat_correl))
        # print("\n")
