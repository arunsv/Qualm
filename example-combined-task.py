import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import sys
from time import time

if __name__ == "__main__":
    
    start = time()
    
    # PMU / MCA stuff
    pcolsS=range(9,28)
    mcolsS=range(30,55)
    
    pcolsSM= range(9,32)
    mcolsSM= range(34,59)  
    
    target_func = 'Mean Cyc/Insn'
    large = 400.0
    #percentiles = [66,68,70,72,74,76,78,80]
    percentiles = [70]
    logTrans=True
        
    fSmallUniq = "./data/Small-Uniq-Stalls-Misses.csv"
    fLargeUniq = "./data/Large-Uniq-Stalls-Misses.csv"
    fLargeDup = "./data/Large-Dup-Stalls-Misses.csv"     
    
    for percentile in percentiles:
        
        dataset = fLargeUniq
        print("Threshold Percentile = ",percentile)
        print("Dataset : ",dataset)
        print("\n")
        pre = pp.PreProcess(dataset, target_func)
        pre.setColumns(icols=range(2,5),pcols=pcolsSM,mcols=mcolsSM)
        X_train,y_train,w_train,X_test,y_test,w_test,cNames,ind = pre.prepareData()
        classes_train, classes_test, thresh = pre.createClassificationData(percentile)
        print("Threshold = ",thresh)
        sp1 = sup.Supervised(X_train,y_train,X_test,y_test,cNames,logTrans=logTrans, cls_train=classes_train,cls_test=classes_test,thresh=thresh)
        sp1.multiStageClassRegNew()
        
        
        #print("Large unique ")
        #print("\n")    
        ##pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
        #pre = pp.PreProcess("./data/stalls-large-unique.csv", target_func)
        #pre.setColumns(icols=range(2,5),pcols=pcols2,mcols=mcols2)
        #X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
        #classes2, thresh = pre.createClassificationData(percentile)
        #print("Threshold = ",thresh)
        #sp2 = sup.Supervised(X2,y2,c_names2,logTrans=logTrans, cls=classes2)
        #sp2.multiStageClassReg()
        
        
        
        #print("Large duplicates ")
        #print("\n")      
        #pre = pp.PreProcess("./data/dataset-large-duplicate-winnow.csv", target_func)
        #pre.setColumns(icols=range(2,5),pcols=pcols3,mcols=mcols3)
        #X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
        #classes3, thresh = pre.createClassificationData(percentile)
        #print("Threshold = ",thresh)
        #sp3 = sup.Supervised(X2,y2,c_names2,logTrans=logTrans,cls=classes3)
        #sp3.multiStageClassReg()
        
        #print("\n") 
        #print("----------------------------------------------------------------------") 
        #print("\n") 
        
    print("Time taken : ", time()-start)
        
    
    


