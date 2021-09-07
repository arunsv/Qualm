import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import sys
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    
    # PMU / MCA stuff
    pcols1=range(42,61)
    mcols1=range(65,90)
    
    pcols2 = range(9,28)
    mcols2 = range(30,55)
    
    pcols3 = range(9,28)
    mcols3 = range(30,55)
    
    target_func = 'Xtra Cyc/Insn'
    
    lo = 0.0
    hi = 4.0

    pre = pp.PreProcess("./data/dataset-small-unique-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols1,mcols=mcols1)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    
    data1 = pre.get_target()
    
    pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols2,mcols=mcols2)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    
    data2 = pre.get_target()
    
    pre = pp.PreProcess("./data/dataset-large-duplicate-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols3,mcols=mcols3)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    
    data3 = pre.get_target()    
    
    plt.figure(figsize=(8,6))
    #plt.hist(data1, bins=100, alpha=0.45, color ='b', label="Small dataset- unique")
    #plt.hist(data2, bins=100, alpha=0.65, color ='y', label="Large dataset - unique")
    #plt.hist(data3, bins=100, alpha=0.25, color ='r', label="Large dataset - duplicates")
    
    #counts1, bins1 = np.histogram(data1, bins=100)
    #counts2, bins2 = np.histogram(data2, bins=100)
    #counts3, bins3 = np.histogram(data3, bins=100)
    
    data = [data1,data2,data3]
    #weights = [np.ones(len(data1))*(len(data1)/), np.ones(len(data2))*(bins2[1]-bins2[0]),np.ones(len(data3))*(bins3[1]-bins3[0])]
    labels = ['Small dataset- unique','Large dataset - unique','Large dataset - duplicates']
    plt.hist(data, bins=100, density = True, histtype='bar', label=labels)
    plt.xlabel("Value of the target function",fontsize=32)
    plt.ylabel("Probability density function value",fontsize=32)
    ax=plt.gca()
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)     
    plt.legend(loc='upper right',fontsize=24) 
    plt.grid()
    plt.show()
    
    
    
    
    


