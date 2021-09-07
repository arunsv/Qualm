import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import seaborn as sns
import imageio
import statsmodels.api as sm
from time import time
from collections import defaultdict
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score

import torch
import torch.nn.functional as F

from ray import tune

class FCNReg(torch.nn.Module):
    def __init__(self, n_feat, n_hid1, n_hid2, n_op, drop = 0.6):
        super().__init__()
        self.ip = torch.nn.Linear(n_feat, n_hid1)   
        self.h1 = torch.nn.Linear(n_hid1, n_hid1)
        self.drop1 = torch.nn.Dropout(drop)
        self.h2 = torch.nn.Linear(n_hid1, n_hid2)
        self.drop2 = torch.nn.Dropout(drop)
        self.h3 = torch.nn.Linear(n_hid2, n_hid2)
        self.op = torch.nn.Linear(n_hid2, n_op)   

    def forward(self, x):
        i1 = F.relu(self.ip(x))     
        h1 = F.relu(self.h1(i1)) 
        d1 = self.drop1(h1)
        h2 = F.relu(self.h2(d1)) 
        d2 = self.drop2(h2)
        h3 = F.relu(self.h3(d2))
        op = self.op(h3)            
        
        return op
    
class FCNClass(torch.nn.Module):
    def __init__(self, n_feat, n_hid1, n_hid2, n_op, drop = 0.5):
        super().__init__()
        self.ip = torch.nn.Linear(n_feat, n_hid1)   
        self.h1 = torch.nn.Linear(n_hid1, n_hid1)
        self.drop1 = torch.nn.Dropout(drop)
        self.h2 = torch.nn.Linear(n_hid1, n_hid2)
        self.drop2 = torch.nn.Dropout(drop)
        self.h3 = torch.nn.Linear(n_hid2, n_hid2)
        self.op = torch.nn.Linear(n_hid2, n_op)   

    def forward(self, x):
        i1 = F.relu(self.ip(x))     
        h1 = F.relu(self.h1(i1)) 
        d1 = self.drop1(h1)
        h2 = F.relu(self.h2(d1)) 
        d2 = self.drop2(h2)
        h3 = F.relu(self.h3(d2))
        op = self.op(h3)            
        
        return op
    
    def predict(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return np.array(ans,dtype=int)    
    
class dlModels():
    
    def __init__(self,X,y,c,logTrans = False, cls = None):
        
        self.X = MinMaxScaler().fit_transform(X)
        self.colNames = c
        self.logTransform = logTrans
        self.y = y
           
        self.alpha = 0.2        
        
        
    def buildFCNRegressor(self):
        
        perfNet = FCNReg(n_feat=self.X.shape[1], n_hid1=20, n_hid2 = 10, n_op=1)
        print(perfNet)
        optimizer = torch.optim.Adam(perfNet.parameters(), lr=0.0077485)
        loss = torch.nn.MSELoss()  
        
        #Training testing split
        x_train, x_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        n_train = x_train.shape[0]
        batch_size = 256
        n_batches = int(n_train/batch_size)
        print("Numberr of samples in trainng set : ", n_train)
        print("Number of batches = ",n_batches, "Total number of samples = ", n_batches*batch_size)
        
        epochs = 3000
        total_loss = []
        final_loss= [0.]*(n_train)
        for i in range(epochs):
            loss_fun = []
            for j in range(n_batches):
                batch = range(j*batch_size,(j+1)*batch_size)
                optimizer.zero_grad()
                y_pred = perfNet(torch.from_numpy(x_train[batch]).float())
                y_train_tensor = torch.from_numpy(y_train[batch]).float().view(batch_size,-1)
                single_loss = loss(y_pred, y_train_tensor)
                final_loss[j] = single_loss
                loss_fun.append(single_loss)
                single_loss.backward()
                optimizer.step()
                
            total_loss.append(sum(loss_fun)/len(loss_fun))   
            if i%100==0:
                print(f'epoch: {i:3} loss: {sum(loss_fun)/len(loss_fun):10.8f}')
        
        #self.plot_loss(total_loss)
        y_val_pred = perfNet(torch.from_numpy(x_val).float())
    
        if (self.logTransform == True):
            self.ytrue = np.exp(y_val) - 1.00
            self.ypred = np.exp(y_val_pred.data.detach().numpy()) - 1.00
        else:
            self.ytrue = y_val
            self.ypred = y_val_pred.data.detach().numpy()
    
       
        if (self.logTransform == True):
            self.ytrue = np.exp(y_val) - 1.00
            self.ypred = np.exp(y_val_pred.data.detach().numpy()) - 1.00
        else:
            self.ytrue = y_val
            self.ypred = y_val_pred.data.detach().numpy()
            
        print("R2 for DL-FCN model : ", r2_score(self.ytrue,self.ypred))                  
        print("RMS error DL-FCN : ", np.sqrt(mean_squared_error(self.ytrue,self.ypred)))           
    

        #y_val_pred = perfNet(torch.from_numpy(x_val).float())
    
        #if (self.logTransform == True):
        #self.ytrue = np.exp(y_val) - 1.00
        #self.ypred = np.exp(y_val_pred.data.detach().numpy()) - 1.00
        #else:
        #self.ytrue = y_val
        #self.ypred = y_val_pred.data.detach().numpy()
    
        ## plot and show learning process
        #plt.cla()
        #ax.set_title('Regression Analysis', fontsize=35)
        #ax.set_xlabel('Ground truth test data', fontsize=24)
        #ax.set_ylabel('Predicted test data', fontsize=24)
        #ax.scatter(self.ytrue, self.ypred)
    
        ## Used to return the plot as an image array 
        ## (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        #fig.canvas.draw()       # draw the canvas, cache the renderer
        #image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        #image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
        #scatter_plots.append(image)
    
        ## save images as a gif    
        #imageio.mimsave('./dlFCN.gif', scatter_plots, fps=10) 
        
        
    def buildFCNClassifier(self):
        
        perfNet = FCNClass(n_feat=self.X.shape[1], n_hid1=20, n_hid2 = 10, n_op=2)
        print(perfNet)
        optimizer = torch.optim.SGD(perfNet.parameters(), lr=0.0040295)
        loss = torch.nn.CrossEntropyLoss()  
        
        #Training testing split
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        n_train = x_train.shape[0]
        batch_size = 64
        n_batches = int(n_train/batch_size)
        print("Numberr of samples in trainng set : ", n_train)
        print("Number of batches = ",n_batches, "Total number of samples = ", n_batches*batch_size)
        
        epochs = 3000
        total_loss = []
        final_loss= [0.]*(n_train)
        for i in range(epochs):
            loss_fun = []
            for j in range(n_batches):
                batch = range(j*batch_size,(j+1)*batch_size)
                optimizer.zero_grad()
                y_pred = perfNet(torch.from_numpy(x_train[batch]).type(torch.FloatTensor))
                y_train_tensor = (torch.from_numpy(y_train[batch]).type(torch.LongTensor))
                y_train_tensor.squeeze_()
                #print(y_pred.size(),y_train_tensor.size())
                single_loss = loss(y_pred, y_train_tensor)
                final_loss[j] = single_loss
                loss_fun.append(single_loss)
                single_loss.backward()
                optimizer.step()
                
            total_loss.append(sum(loss_fun)/len(loss_fun))   
            if i%100==0:
                print(f'epoch: {i:3} loss: {sum(loss_fun)/len(loss_fun):10.8f}')
        
        self.plot_loss(total_loss)
        l_pred = perfNet.predict(torch.from_numpy(x_test).float())

        #correct_results_sum = (l_pred == y_test).sum().float()
        #acc = correct_results_sum/y_test.shape[0]
        #acc = torch.round(acc * 100)  
        #print("Percentage accuracy : ",acc)
                 
        conf_mat = confusion_matrix(y_test,l_pred)
        print("Confusion matrix for the current model : ")
        print(conf_mat)
        print("Classification accuracy : ", accuracy_score(y_test,l_pred)) 
        print("Precision score = ",precision_score(y_test,l_pred))
        print("Recall score = ",recall_score(y_test,l_pred))
        print("F1 score = ",f1_score(y_test,l_pred))  
        
        
    def plot_loss(self,loss):
        
        plt.figure()
        plt.plot(loss,linewidth = 3.0, marker ='+', mS = 2)
        plt.xlabel('Epochs', fontsize = 20)
        plt.ylabel('Loss', fontsize = 20)
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)    
        #plt.legend(fontsize=22)
        plt.grid()
        plt.show()        
        
        
    def buildFCNClassifierForTuning(self,lrn_rate,drp_out):
        
        perfNet = FCNClass(n_feat=self.X.shape[1], n_hid1=20, n_hid2 = 10, n_op=2, drop=drp_out)
        print(perfNet)
        optimizer = torch.optim.SGD(perfNet.parameters(), lr=lrn_rate)
        loss = torch.nn.CrossEntropyLoss()  
        
        #Training testing split
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        n_train = x_train.shape[0]
        batch_size = 64
        n_batches = int(n_train/batch_size)
        print("Numberr of samples in trainng set : ", n_train)
        print("Number of batches = ",n_batches, "Total number of samples = ", n_batches*batch_size)
        
        epochs = 3000
        total_loss = []
        final_loss= [0.]*(n_train)
        for i in range(epochs):
            loss_fun = []
            for j in range(n_batches):
                batch = range(j*batch_size,(j+1)*batch_size)
                optimizer.zero_grad()
                y_pred = perfNet(torch.from_numpy(x_train[batch]).type(torch.FloatTensor))
                y_train_tensor = (torch.from_numpy(y_train[batch]).type(torch.LongTensor))
                y_train_tensor.squeeze_()
                #print(y_pred.size(),y_train_tensor.size())
                single_loss = loss(y_pred, y_train_tensor)
                final_loss[j] = single_loss
                loss_fun.append(single_loss)
                single_loss.backward()
                optimizer.step()
                
            total_loss.append(sum(loss_fun)/len(loss_fun))   
            if i%100==0:
                print(f'epoch: {i:3} loss: {sum(loss_fun)/len(loss_fun):10.8f}')
        
        #self.plot_loss(total_loss)
        l_pred = perfNet.predict(torch.from_numpy(x_test).float())
        f1 = recall_score(y_test,l_pred)
        return f1
    
    
        
