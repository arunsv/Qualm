import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from time import time
from collections import defaultdict
from pprint import pprint
import sys

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn.cluster import AffinityPropagation,SpectralClustering
from sklearn.cluster import KMeans, DBSCAN


from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

class UnSupervised():
    
    def __init__(self,X,y,c):
        
        self.X = MinMaxScaler().fit_transform(X)
        self.y = y
        self.colNames = c
        
    def doPCA(self,X):
        
    
        #Remove the zero variance columns
        selector = VarianceThreshold()
        selector.fit_transform(self.X)    
        
        # Standardizing the features
        x = StandardScaler().fit_transform(self.X)   
        pca = PCA(n_components=40)
        pC = pca.fit_transform(x)
    
        sigma = pca.explained_variance_ratio_
        sigExp = np.zeros((len(sigma)))
        for idx in range(len(sigma)):
            sigExp[idx] = np.sum(sigma[:idx])
            
        return sigExp,pC    
    

    def plotPCA(self,sigExp):
        
        plt.figure()
        plt.plot(np.linspace(1,len(sigExp),len(sigExp),dtype=int),100.*sigExp,marker = 'o', mS = 12)
        plt.grid()
        plt.title('Variance explained by the principal components',fontsize = 25)
        plt.xlabel('Number of principal components',fontsize=20)
        plt.ylabel('Percentage variance explained',fontsize=20)
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15) 
        plt.show()    
        
    def clustering(self):
        
        
        df = pd.DataFrame(self.X)
        #df.columns = cols
        corrs = df.corr()
        print(corrs.shape)
        print(np.round(corrs,2))
        
        #Clustermap using seaborn
        #corrs_nonan = np.nan_to_num(corrs.values)
        #sns.set(color_codes=True)
        #sns.clustermap(corrs,figsize=(13, 13))
        #plt.savefig("clus.png")
        
        #Try different clustering methods
    
        clus = KMeans(n_clusters=8, random_state=0).fit(np.transpose(self.X))
        f = open("./clustering_pmu_kmeans.txt","w")
        f.write("Clustering info (k-means (8 clusters requested) with Euclidean distance) : \n")
        
        #clus = AffinityPropagation(affinity='precomputed').fit(corrs)
        #f = open("./clustering_pmu_corr.txt","w")
        #f.write("Clustering info (Affinity Propagation with correlation based similarity) : \n")
        
        #clus = AffinityPropagation().fit(np.transpose(X))
        #f = open("./clustering_pmu_dist.txt","w")    
        #f.write("Clustering info (Affinity Propagation with negative squared Euclidean distance based similarity) : \n")
        
        clusters = clus.labels_
        cluster_maps = defaultdict(list)
        _count = 0
        for _id in clusters:
            cluster_maps[_id].append(self.colNames[_count])
            _count+=1
        
        
        for _id in cluster_maps:
            
            if (len((cluster_maps[_id])) >= 2):
                
                f.write("--------------------------------------------------------\n")
                f.write(str("Cluster ID : "+ str(_id+1)))
                f.write("\n")
                f.write(str(cluster_maps[_id]))
                f.write("\n--------------------------------------------------------\n")
            
        f.close()
        
        
    
    def featureClustering(self):
        
        #X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.5, random_state=0)
        
        #self.X = self.X[self.y < 10.0]
        print("Total training data : ", self.X.shape[0])
        #clus = SpectralClustering(n_clusters=3).fit(X_train)
        
        #length = np.sqrt((self.X**2).sum(axis=1))[:,None]
        #self.X = self.X / length
        self.X = self.X[:,[0,2,3,4]]
        clus = KMeans(n_clusters=3, random_state=0).fit(self.X)
        #clus = DBSCAN(eps=0.6, min_samples=10).fit(X_train)
        #clus = AffinityPropagation().fit(X_train)
            
        labels = clus.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)    
        
        for idx in range(n_clusters_):
            print("Label = ",idx, " Label Count = ", len(np.where(labels == idx)[0]))          
        
        plt.figure()
        
        for idx in range(n_clusters_):
            indices = np.where(labels == idx)[0]
            sns.kdeplot(self.y[indices],label=("Label = "+str(idx)))
            #plt.subplot(n_clusters_,1,idx+1)
            #plt.hist(self.y[indices],bins=20)
            plt.title(("Label = "+str(idx)))
 
        plt.xlabel('Target Value',fontsize=22)
        plt.ylabel('PDF value',fontsize=22)
        
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)    
        plt.legend(fontsize=22)
        plt.grid()
        plt.show()        
        plt.show()        