import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    
    # PMU / MCA stuff
    pcolsSM= range(9,32)
    mcolsSM= range(34,59)      
    
    target_func = 'Xtra Cyc/Insn'
    
    lo = 0.0
    hi = 400.0

    pre = pp.PreProcess("./data/Small-Uniq-Stalls-Misses.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcolsSM,mcols=mcolsSM)
    pre.setLimits(lo, hi)
    X_train,y_train,w_train,X_test,y_test,w_test,cNames,ind = pre.prepareData()
    sp1 = sup.Supervised(X_train,y_train,X_test,y_test,cNames)
    feat1,correl1 = sp1.correlationAnalysis(k=15)
    
    
    pre = pp.PreProcess("./data/Large-Uniq-Stalls-Misses.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcolsSM,mcols=mcolsSM)
    pre.setLimits(lo, hi)
    X_train,y_train,w_train,X_test,y_test,w_test,cNames,ind = pre.prepareData()
    sp2 = sup.Supervised(X_train,y_train,X_test,y_test,cNames)
    feat2,correl2 =sp2.correlationAnalysis(k=15)    
    
   
    pre = pp.PreProcess("./data/Large-Dup-Stalls-Misses.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcolsSM,mcols=mcolsSM)
    pre.setLimits(lo, hi)
    X_train,y_train,w_train,X_test,y_test,w_test,cNames,ind = pre.prepareData()
    sp3 = sup.Supervised(X_train,y_train,X_test,y_test,cNames)
    feat3,correl3 =sp3.correlationAnalysis(k=15)
    
    
    df = pd.DataFrame(list(zip(feat1,correl1,feat2,correl2,feat3,correl3)), columns=['Ft-sm-u','Cl-sm-u','Ft-lg-u','Cl-lg-u','Ft-lg-d','Cl-lg-d'])
    df.to_csv("correls_new.csv",index=False)