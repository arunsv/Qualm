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
    
    target_func = 'Xtra Cyc/Insn'
    large = 200.0
    logTrans=False
        
    fSmallUniq = "./data/Small-Uniq-Stalls-Misses.csv"
    fLargeUniq = "./data/Large-Uniq-Stalls-Misses.csv"
    fLargeDup = "./data/Large-Dup-Stalls-Misses.csv"    
    
    #fileNames = [fSmallUniq, fLargeUniq, fLargeDup]
    #fileDesc = ["Small-uniq","Large-uniq","Large-dup"]   
    
    fileNames = [fSmallUniq, fLargeUniq]
    fileDesc = ["Small-uniq","Large-uniq"]
    
    start = time()
    
    for fileName,descrp in zip(fileNames,fileDesc):
        
        print("Dataset : ", fileName, " ",descrp)
        print("\n")
        pre = pp.PreProcess(fileName, target_func)
        pre.setColumns(icols=range(2,5),pcols=None,mcols=mcolsSM)
        X_train,y_train,w_train,X_test,y_test,w_test,cNames,ind = pre.prepareData()
        sp = sup.Supervised(X_train,y_train,X_test,y_test,cNames,logTrans=logTrans)
        sp.rf_models_CV_Reg()
        print("\n\n")
        
        
        
    print("Time taken : ", time()-start)





