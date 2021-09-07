import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from collections import defaultdict, OrderedDict
import sys
from scipy.stats import percentileofscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class PreProcess():
    
    def __init__(self,srcFile,objFunc,skipRows=1,test_flag = True):
        
        self.srcFile = srcFile
        self.lFlag = False
        self.obj = objFunc
        self.srows = skipRows
        self.tFlag = test_flag

        
        self.X_train = None
        self.y_train = None
        self.w_train = None
        self.X_test = None
        self.y_test = None
        self.w_test = None        
        self.c = None
        self.ind = None
        

       
    def setLimits(self,L,U):
        
        self.lFlag = True
        self.Low = L
        self.Hi = U
    
    def setColumns(self, icols = None, pcols = None, mcols = None):
        
        self.idCols = icols
        self.pmuCols = pcols
        self.mcaCols = mcols
        
    def prepareData(self,topdown = None):
        
        print("Reading Excel file....")
        start = time()
        df = None
        
        if ((self.srcFile).endswith('.xlsx')):
            df = pd.read_excel(self.srcFile,skiprows=self.srows)
        elif ((self.srcFile).endswith('.csv')):
            df = pd.read_csv(self.srcFile) 
            
        print("Read Excel file in ", round(time()-start,3), " seconds.")
        
        if (self.lFlag == True):
            
            print(len(df[df[self.obj] < self.Low]))
            print(len(df[((df[self.obj] > self.Low) & (df[self.obj] < self.Hi))]))
            print(len(df[df[self.obj] > self.Hi]))
            
            df=df[((df[self.obj] > self.Low) & (df[self.obj] < self.Hi))]

        #Do normalization by number of instructions here
        #df = df.dropna(axis = 'index', how='any')
        self.df_full = df
        ind = df['Order/New'].values
        dfw = df.iloc[:,self.idCols]
        
        if (self.pmuCols):
            print("Collecting PMU features and normalizing by PMU Insn.")
            dfx_p = df.iloc[:,self.pmuCols].div(df['LBR Insn'], axis=0)
            #dfx_p = df.iloc[:,self.pmuCols]
            dfx = dfx_p
        if (self.mcaCols):
            print("Collecting MCA features.")
            dfx_m = df.iloc[:,self.mcaCols]
            dfx = dfx_m
            
        if ((self.pmuCols) and (self.mcaCols)):
            print("Concatenating normalized PMU columns and MCA columns.")
            dfx = pd.concat([dfx_p, dfx_m], axis=1)
        
        dfy = df[self.obj]
        #dfy = df[[self.obj,'LBR Insn']]
        x_shape = dfx.shape
        y_shape = dfy.shape
       
        dfw['Name'] = dfw['Superblock Module']+"---"+dfw['Start IP']+"-"+dfw['End IP']
        #dfw['Name'] = dfw['Superblock Module']
        
        print("Dropping NaN columns.")
        nan_count = dfx.isnull().sum(axis = 0)
        drop_cols = []
        for col in nan_count.index:
            if (nan_count[col] > 28000):
                print("Dropped Col: ",col, " NaN count : ", nan_count[col])
                drop_cols.append(col)
        
        dfx = dfx.drop(columns=drop_cols)
        colNames = np.array(list(dfx.columns.values.tolist()))
        
        X = dfx.values
        y = dfy.values
        w = dfw['Name'].values.tolist()
        print("X and y shapes : ",np.shape(X), np.shape(y))    
       
        #Check all-zero rows in X and filter y=w and y correspondingly
        print("Checking all-zero PMU rows.")
        all_zind = np.where(~X.any(axis=1))[0]
        print(len(all_zind), " rows have all zero features out of ",X.shape[0])
        X = np.delete(X,all_zind,axis=0)  
        y = np.delete(y,all_zind) #For thrpt. Check this ,axis=0
        w = np.delete(w,all_zind)
        ind = np.delete(ind,all_zind)
        self.w = w
        
        print("Checking NaN PMU rows.")
        nan_ind = np.where(np.isnan(X).any(axis=1))[0]
        print(len(nan_ind), " rows have some NaN features out of ",X.shape[0])
        X = np.delete(X,nan_ind,axis=0)
        y = np.delete(y,nan_ind) #For thrpt. Check this ,axis=0
        w = np.delete(w,nan_ind)  
        ind = np.delete(ind,nan_ind) 
        self.y = y
        
        print("Checking all-zero PMU columns.")
        all_zind_cols = np.where(~X.any(axis=0))[0]
        print(len(all_zind_cols), " columns have all zero features out of ",X.shape[1])
        X1 = np.delete(X,all_zind_cols,axis=1)  
        newColNames = np.delete(colNames,all_zind_cols)
           
        print("Dimensions of original data : ", x_shape,y_shape)
        #print(dfx.head())      
        #print(dfy.head())    
        print('Dimensions of feature matrix and target : ',X.shape,y.shape)
        print("Any NaNs in features : ", np.isnan(X).any())
        print("Any NaNs in target : ", np.isnan(X).any())
        print("Target min / max : ", np.min(y), " ",np.max(y))
        
        
        X = MinMaxScaler().fit_transform(X1)
        
        if (self.tFlag):
            
            X_train, X_test, y_train, y_test, w_train, w_test, ind_train, ind_test = train_test_split(X, y, w, ind, test_size=0.2, random_state=1)  
            
            self.X_train = X_train
            self.y_train = y_train
            self.w_train = w_train
            self.X_test = X_test
            self.y_test = y_test
            self.w_test = w_test        
            self.c = newColNames
            self.ind = ind
            
            if not (topdown is None):
                    
                tp = self.df_full[df['Order/New'].isin(ind_test)]
                tplist = np.array((tp[topdown].values.tolist()))
                tplist = np.nan_to_num(tplist,nan=0.0,posinf=0.0,neginf=0.0)
                self.ind = tplist
            
            #print("\nTop down prediction")
            #print(self.ind)
            
        else:
            
            self.X_train = X
            self.y_train = y
            self.w_train = w
            self.c = newColNames
            self.ind = ind
        
        return self.X_train,self.y_train,self.w_train,self.X_test,self.y_test,self.w_test,self.c,self.ind        
    
    def getThresholdedData(self,X,y,thresh = 0.0):
        
        Xt = None
        yt = None
        
        if (self.lFlag == True):
            
            Xt = X[((y > self.Low) & (y < self.Hi))]
            yt = y[((y > self.Low) & (y < self.Hi))]
        
        else:
            
            print("Using the provided threshold.")
            
            Xt = X[(y > thresh)]
            yt = y[(y > thresh)]
        
        print("Feature matrix shape = ",Xt.shape)
        return Xt,yt
    
    def plotAvgCPIRanges(self):
        
        cpi_vals = defaultdict(list)
        cpi_stats = dict()
        print("Number of data points : ", len(self.w_train), " ", len(self.y))
        for idx in range(len(self.w_train)):
            cpi_vals[self.w_train[idx]].append(self.y[idx])
        
        sblks = []
        sblks.append('ntoskrnl.exe---0x1401c93a0-0x1401c93d0')
        sblks.append('dxgkrnl.sys---0x1c0007112-0x1c0007147')        
        sblks.append('chrome.dll---0x180001176-0x180001183')
        sblks.append('mfeaack.sys---0x140037e92-0x140037eaa')
        sblks.append('rtmpltfm.dll---0x10307322-0x10307327')
        sblks.append('google play music desktop player.exe---0x17b3914-0x17b391f')
        
        all_cpi_mca = [0.3636, 0.2, 0.2167,0.2, 0.2, 0.16]
        
        new_w = []
        new_y = []

        for key,vals in cpi_vals.items():
            if (key in sblks):
                print(key,vals)
                sep = '---'
                key_new = key.split(sep, 1)[0]
                if (key_new == 'google play music desktop player.exe'):
                    key_new = 'google music.exe'
                key_new=key_new[:-4]
                w_ext = [key_new]*len(vals)
                new_w.extend(w_ext)
                new_y.extend(vals)
   
         #for key,vals in cpi_vals.items():
            
            #if (len(vals) > 400):
                ##print(key, " : ",len(vals))
                #min_cpi = np.min(vals)
                #med_cpi = np.median(vals)
                #max_cpi = np.max(vals)
                #mean_cpi = np.mean(vals)
                #std_cpi = np.std(vals)
                #stat_tup = tuple((mean_cpi,std_cpi,min_cpi,med_cpi,max_cpi))
                #cpi_stats[key] = stat_tup
        
                
        #print(len(cpi_stats), " super blocks")
        ##Prepare data frame
        d = {'Superblock':new_w,'Avg CPI':new_y}
        df = pd.DataFrame(d)
        print(df.columns, df.shape)
        b = sns.boxplot(x=df['Superblock'],y=df['Avg CPI'])
        b.set_xlabel("App/Superblock name",fontsize=40)
        b.set_ylabel("Avg. CPI",fontsize=40)
        b.tick_params(axis='x',labelsize=30)
        b.tick_params(axis='y',labelsize=32)
        plt.scatter(sblks,all_cpi_mca,c='r',s=180,label = 'CPI-MCA-min')
        plt.legend(fontsize=36)
        plt.tight_layout()
        plt.show()
        
        #sns.barplot(x=df['Superblock Name'],y=df['Avg CPI'])
        
        #labels, data = [*zip(*cpi_new.items())]
        #plt.boxplot(data)
        #plt.xticks(range(1, len(labels) + 1), labels)
        plt.show()
        
    def draw_histogram(self,nBins):
    
        plt.figure()
        plt.hist(self.y,bins=nBins,density=True)
        #plt.axvline(x=0.79,color='k',linewidth=3.0)
        plt.grid()
        #plt.title('Histogram of the target function  ',fontsize = 36)
        plt.xlabel('Value of the target function ',fontsize=32) #r'$\frac{\sigma}{\mu}$'
        plt.ylabel('Probability density function value',fontsize=32)
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=22)
        ax.yaxis.set_tick_params(labelsize=22)     
        plt.show()
    
    def get_target(self):
        
        return self.y
    
    def distance_analysis(self,pmu,pmu_u,tar,sb_names,ind):
        
        print("Computing pair-wise distances.")
        distances = {}
        for _idx in range(pmu.shape[0]):
            for _jdx in range(_idx+1,pmu.shape[0]):
                
                if ((np.linalg.norm(pmu[_idx,:],1) > 0.0) and (np.linalg.norm(pmu[_jdx,:],1) > 0.0)):
                    
                    tup = tuple((_idx,_jdx))
                    dist = np.linalg.norm((pmu[_idx,:]-pmu[_jdx,:]),ord=2)
                    distances[tup] = dist
        
        all_dist = distances.values()
        #all_dist_sorted = sorted(all_dist)
        thresh = 0.001
        t_dist = [x for x in all_dist if x < thresh]
        print("Average pair-wise distance : ",np.mean(np.array(list(all_dist))))
        print("Average pair-wise std. dev : ",np.std(np.array(list(all_dist))))
        print("Number of pairs with distances below threshold : ", len(t_dist), " out of ", len(all_dist))
        count = 0
        for (pmu_pair,dist) in distances.items():
            if (dist < thresh):
                count+=1
                _idx = pmu_pair[0]
                _jdx = pmu_pair[1]
                print("")
                print("Pair number : ",count)
                print("Index 1 : ",ind[_idx]," Super Block 1 : ",sb_names[_idx])
                print("PMU-1 : ", np.round(pmu_u[_idx,:],5)) 
                print("Index 2 : ",ind[_jdx]," Super Block 2 : ",sb_names[_jdx])
                print(" PMU-2 : ", np.round(pmu_u[_jdx,:],5))
                print("PMU-space Euclidean Distance : ", np.round(dist,6), " Target 1 : ",np.round(tar[_idx],4)," Target 2 : ", np.round(tar[_jdx],4))
                print("")
        
        #dist_np = np.array(list(distances.values()),dtype=float)
        #counts, bin_edges = np.histogram(dist_np, bins=100, density=True)
        #cdf = np.cumsum(counts)
        #plt.plot(bin_edges[1:], cdf)
        #plt.title('CDF of the pair-wise distances  ',fontsize = 25)
        #plt.xlabel('Value of the distance ',fontsize=20) #r'$\frac{\sigma}{\mu}$'
        #plt.ylabel('Cumulative density function value',fontsize=20)    
        #ax=plt.gca()
        #ax.xaxis.set_tick_params(labelsize=15)
        #ax.yaxis.set_tick_params(labelsize=15)     
        #plt.show()
    
        threshes = [0.0001,0.001,0.01]
        for thresh in threshes:
            t_dist = [x for x in all_dist if x < thresh]
            per = len(t_dist)*100./float(len(all_dist))
            print("Threshold value : ", thresh, " Percentage distances under threshold : ", round(per,5))
        
        plt.figure()
        plt.hist(all_dist,bins=100,density=True)
        plt.grid()
        plt.title('Histogram of all-pair-wise distaces',fontsize = 25)
        plt.xlabel('Value of the distance ',fontsize=20) #r'$\frac{\sigma}{\mu}$'
        plt.ylabel('Probability density function value',fontsize=20)
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)     
        plt.show()     
        
        
    def createClassificationData(self,per):
        
        #self.y = self.y[:,0]
        print(self.y_train.shape)
        thresh = np.percentile(self.y_train,per)
        print("Threshold : ", thresh)
        
        classes_train = ((self.y_train > thresh)+np.zeros(len(self.y_train)))
        classes_train = np.array([int(c) for c in classes_train])
        self.classes_train = classes_train
        
        classes_test = ((self.y_test > thresh)+np.zeros(len(self.y_test)))
        classes_test = np.array([int(c) for c in classes_test])
        self.classes_test = classes_test        
        
        #print ("Total number of ones : ", np.sum(self.classes))
        #print ("Total number of samples : ", len(self.classes))
        
        return (self.classes_train, self.classes_test, thresh)
        
    def plotClassImbalance(self):
        
        thresh = np.linspace(0.5,2.0,16)
        percentiles = np.zeros(16)
        for idx in range(16):
            percentiles[idx] = 100.0-percentileofscore(self.y,thresh[idx])
            
        plt.figure()
        plt.plot(thresh,percentiles,linewidth = 3, marker = 'o', mS =12)
        plt.grid()
        plt.title('Percentage of bottleneck samples with \n varying target threshold',fontsize = 25)
        plt.xlabel('Value of the target function threshold ',fontsize=20) #r'$\frac{\sigma}{\mu}$'
        plt.ylabel('Percentage of the bottleneck samples',fontsize=20)
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)     
        plt.show()         
        
        
    def write_fp_fn_data(self, ind_gl, thresh, ind_fp, ind_fn, y_vals_fp, y_vals_fn, top_cols):
        
               
        order_fp = [ind_gl[idx] for idx in ind_fp]
        order_fn = [ind_gl[idx] for idx in ind_fn]
        
        diff_fp = (thresh - y_vals_fp)
        diff_fn = (y_vals_fn - thresh)
        
        dict_fp = dict(zip(order_fp, diff_fp))
        dict_fn = dict(zip(order_fn, diff_fn))
        
        reorder_fp = [x for _,x in sorted(zip(diff_fp,order_fp),reverse=True)]
        reorder_fn = [x for _,x in sorted(zip(diff_fn,order_fn),reverse=True)]
        
        reorder_fp_new = reorder_fp[:15]
        reorder_fn_new = reorder_fn[:15]
        
        #print(reorder_fp_new)
        #print(reorder_fn_new)
        
       
        df_fp_full = pd.DataFrame(columns=self.df_full.columns)
        df_fn_full = pd.DataFrame(columns=self.df_full.columns)

         
        for idx in range(len(reorder_fp_new)):
            extracted_df = self.df_full[self.df_full['Order/New'] == reorder_fp_new[idx]]
            df_fp_full = df_fp_full.append(extracted_df)
            
            #print(extracted_df[['Order/New',self.obj]])
            #print(df_fp_full[['Order/New',self.obj]])
            #print("\n")
            #print(extracted_df[self.obj], dict_fp[reorder_fp_new[idx]])
        
        #print("\n\n")
        
        for idx in range(len(reorder_fn_new)):
            extracted_df = self.df_full[self.df_full['Order/New'] == reorder_fn_new[idx]]
            df_fn_full = df_fn_full.append(extracted_df)
            
            #print(extracted_df[['Order/New',self.obj]])
            #print(df_fn_full[['Order/New',self.obj]])
            #print("\n")
            #print(extracted_df[self.obj], dict_fn[reorder_fn_new[idx]])
        
        #print(df_fp[['Order/New',self.obj]])
        #print(df_fn[['Order/New',self.obj]])
        
        pmu_feat, mca_feat = self.get_additional_columns()
        print("Total new columns : ",(len(pmu_feat)+len(mca_feat)))
        cols_all = self.df_full.columns.tolist()
        cols_id = [cols_all[idx] for idx in self.idCols]
        col_obj = [self.obj]
        
                
        df_fp_full['Blank'] = ""
        frames1 = [df_fp_full[cols_id],df_fp_full['Blank'],df_fp_full[col_obj],df_fp_full['Blank'],df_fp_full[top_cols]]
        frames2 = [df_fp_full['Blank'],df_fp_full[pmu_feat],df_fp_full['Blank'],df_fp_full[mca_feat]]
        df_fp = pd.concat((frames1+frames2),axis=1)

        df_fn_full['Blank'] = ""
        frames1 = [df_fn_full[cols_id],df_fn_full['Blank'],df_fn_full[col_obj],df_fn_full['Blank'],df_fn_full[top_cols]]
        frames2 = [df_fn_full['Blank'],df_fn_full[pmu_feat],df_fn_full['Blank'],df_fn_full[mca_feat]]
        df_fn = pd.concat((frames1+frames2),axis=1)        
        
        df_fp.to_csv("fp_new.csv", index=False)
        df_fn.to_csv("fn_new.csv", index=False)
        
                
    def getTDData(self,colD,colS,colG,colM):
        
        print("Reading Excel file....")
        start = time()
        df = None
        
        if ((self.srcFile).endswith('.xlsx')):
            df = pd.read_excel(self.srcFile,skiprows=self.srows)
        elif ((self.srcFile).endswith('.csv')):
            df = pd.read_csv(self.srcFile) 
            
        print("Read Excel file in ", round(time()-start,3), " seconds.")
        
        if (self.lFlag == True):
            
            print(len(df[df[self.obj] < self.Low]))
            print(len(df[((df[self.obj] > self.Low) & (df[self.obj] < self.Hi))]))
            print(len(df[df[self.obj] > self.Hi]))
            
            df=df[((df[self.obj] > self.Low) & (df[self.obj] < self.Hi))]

        #Do normalization by number of instructions here
        #df = df.dropna(axis = 'index', how='any')
        self.df_full = df
        ind = df['Order/New'].values
        dfw = df.iloc[:,self.idCols]
        
        if (self.pmuCols):
            print("Collecting PMU features and normalizing by PMU Insn.")
            dfx_p = df.iloc[:,self.pmuCols].div(df['LBR Insn'], axis=0)
            #dfx_p = df.iloc[:,self.pmuCols]
            dfx = dfx_p
        if (self.mcaCols):
            print("Collecting MCA features.")
            dfx_m = df.iloc[:,self.mcaCols]
            dfx = dfx_m
            
        if ((self.pmuCols) and (self.mcaCols)):
            print("Concatenating normalized PMU columns and MCA columns.")
            dfx = pd.concat([dfx_p, dfx_m], axis=1)
        
        dfy = df[list([colD,colS,colG,colM])]
        x_shape = dfx.shape
        y_shape = dfy.shape
       
         
        dfw['Name'] = dfw['Superblock Module']+"---"+dfw['Start IP']+"-"+dfw['End IP']
        #dfw['Name'] = dfw['Superblock Module']
        
        print("Dropping NaN columns.")
        nan_count = dfx.isnull().sum(axis = 0)
        drop_cols = []
        for col in nan_count.index:
            if (nan_count[col] > 28000):
                print("Dropped Col: ",col, " NaN count : ", nan_count[col])
                drop_cols.append(col)
        
        dfx = dfx.drop(columns=drop_cols)
        colNames = np.array(list(dfx.columns.values.tolist()))
        
        X = dfx.values
        y = dfy.values
        w = dfw['Name'].values.tolist()
        print("X and y shapes : ",np.shape(X), np.shape(y))    
       
        #Check all-zero rows in X and filter y=w and y correspondingly
        print("Checking all-zero PMU rows.")
        all_zind = np.where(~X.any(axis=1))[0]
        print(len(all_zind), " rows have all zero features out of ",X.shape[0])
        X = np.delete(X,all_zind,axis=0)  
        y = np.delete(y,all_zind,axis=0) #For thrpt. Check this ,axis=0
        w = np.delete(w,all_zind)
        ind = np.delete(ind,all_zind)
        
        print("Checking NaN PMU rows.")
        nan_ind = np.where(np.isnan(X).any(axis=1))[0]
        print(len(nan_ind), " rows have some NaN features out of ",X.shape[0])
        X = np.delete(X,nan_ind,axis=0)
        y = np.delete(y,nan_ind,axis=0) #For thrpt. Check this ,axis=0
        w = np.delete(w,nan_ind)  
        ind = np.delete(ind,nan_ind) 
        self.y = y
        
        print("Checking all-zero PMU columns.")
        all_zind_cols = np.where(~X.any(axis=0))[0]
        print(len(all_zind_cols), " columns have all zero features out of ",X.shape[1])
        X1 = np.delete(X,all_zind_cols,axis=1)  
        newColNames = np.delete(colNames,all_zind_cols)
           
        print("Dimensions of original data : ", x_shape,y_shape)
        #print(dfx.head())      
        #print(dfy.head())    
        print('Dimensions of feature matrix and target : ',X.shape,y.shape)
        print("Any NaNs in features : ", np.isnan(X).any())
        print("Any NaNs in target : ", np.isnan(y).any())
                
        
        X = MinMaxScaler().fit_transform(X1)
        
        if (self.tFlag):
            
            X_train, X_test, y_train, y_test, w_train, w_test, ind_train, ind_test = train_test_split(X, y, w, ind, test_size=0.2, random_state=1)  
            
            self.X_train = X_train
            self.y_train = y_train
            self.w_train = w_train
            self.X_test = X_test
            self.y_test = y_test
            self.w_test = w_test        
            self.c = newColNames
            self.ind = ind
            
            print("TD shape before sanitizing: ", self.y_test.shape)
            mask = np.any((np.isnan(self.y_test) | np.equal(self.y_test, 0.0)| np.greater_equal(self.y_test, 1000.0)), axis=1)
            self.y_test = self.y_test[~mask]            
            print("Final TD shape : ", self.y_test.shape)

           
        else:
            
            self.X_train = X
            self.y_train = y
            self.w_train = w
            self.c = newColNames
            self.ind = ind
        
        return self.y_test        
    
    def getThresholdedData(self,X,y,thresh = 0.0):
        
        Xt = None
        yt = None
        
        if (self.lFlag == True):
            
            Xt = X[((y > self.Low) & (y < self.Hi))]
            yt = y[((y > self.Low) & (y < self.Hi))]
        
        else:
            
            print("Using the provided threshold.")
            
            Xt = X[(y > thresh)]
            yt = y[(y > thresh)]
        
        print("Feature matrix shape = ",Xt.shape)
        return Xt,yt    
    
        
    def get_additional_columns(self):
        
        colH_PMU = OrderedDict( [
            # Frontend
            ('FRONTEND_RETIRED.L1I_MISS_PS',              'l1i-miss'),
            ('FRONTEND_RETIRED.ITLB_MISS_PS',             'itlb-miss'),
            ('BACLEARS.ANY',                              'baclear'),
            ('FRONTEND_RETIRED.DSB_MISS_PS',              'dsb-miss'),
        
            # Bad Spec
            ('BR_MISP_RETIRED.ALL_BRANCHES_PS',           'br-misp'),
            
            # Backend
            ('MEM_LOAD_RETIRED.FB_HIT_PS',                'fb-hit'),
            ('MEM_LOAD_RETIRED.L1_MISS_PS',               'l1-miss'),
            ('DTLB_LOAD_MISSES.STLB_HIT',                 'stlb-hit-ld'),
            ('MEM_INST_RETIRED.STLB_MISS_LOADS_PS',       'stlb-miss-ld'),
            ('DTLB_STORE_MISSES.STLB_HIT',                'stlb-hit-st'),
            ('MEM_INST_RETIRED.STLB_MISS_STORES_PS',      'stlb-miss-st'),
            ('MEM_LOAD_RETIRED.L2_MISS_PS',               'l2-miss'),
            ('MEM_LOAD_L3_HIT_RETIRED.XSNP_HITM_PS',      'l3-hitm'),
            ('MEM_LOAD_RETIRED.L3_MISS_PS',               'l3-miss'),
            ('PARTIAL_RAT_STALLS.SCOREBOARD',             'rat-stalls'),
            ('EXE_ACTIVITY.BOUND_ON_STORES',              'st-bnd'),
            ('MACHINE_CLEARS.COUNT',                      'clears'),
            ('ARITH.DIVIDER_ACTIVE',                      'div'),
            
            # Retire
            ('IDQ.MS_UOPS',                               'ms-uops'),
        ] )
        
        colH_PMUx = { x : None  for x in colH_PMU.values() }
        colL_PMU_shrt = colH_PMUx.keys()

        colL_MCA = [
            'MCA-min',
            'MCA-sim',
            'CPI-LBR-min',
            'CPI-LBR-med',
            'CPI-MCA-min',
            'CPI-MCA-sim',
            'Xcpi-Wd-min',
            'Xcpi-Wd-sim',
            'Lat-Wd',
            'Lat-Wp',
            'Lat-E',
            'Lat-Wr',
            'LatT',
            'Lat0',
            'Lat1',
            'StallP-RAT',
            'StallP-RCU',
            'StallP-SCHEDQ',
            'StallP-LQ',
            'StallP-SQ',
            'StallP-GROUP',
            'Resource Pressure',
            'Register Dependencies',
            'Memory Dependencies',
            'Cycles w Backend Pressure',
            'Ld',
            'St' ]
        
        return (colL_PMU_shrt,colL_MCA)
        