import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import sys

if __name__ == "__main__":
    

    # PMU / MCA cols
    
    #Small-Unique
    #pcols=range(42,61)
    #mcols=range(65,90)
    
    #Large - Unique and Duplicate
    pcols = range(9,28)
    mcols = range(30,55)
    
    
    #target_func = 'Xtra Cyc/Insn'
    target_func = 'Mean Cyc/Insn'
    lo = 0.0
    hi = 500.0

    #pre = pp.PreProcess("./data/dataset-small-unique-winnow.xlsx", target_func)
    pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols,mcols=mcols)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    print("Target function : ", target_func)
    print("Min = ", np.min(y2), " Med = ", np.median(y2), " Max = ", np.max(y2), " Mean = ",np.mean(y2), " Std = ",np.std(y2) )
    #classes, thresh = pre.createClassificationData(90)    

    #sp = sup.Supervised(X2,y2,c_names2,logTrans = False,cls=classes)
    #sp.regression_allModels()
    
    #fp_inds,fn_inds,y_vals_fp,y_vals_fn,top_feat = sp.get_fp_fn()
    #pre.write_fp_fn_data(indices2, thresh, fp_inds, fn_inds, y_vals_fp, y_vals_fn,top_feat)

    #pre1 = pp.PreProcess("../data/data-mca-pmu-v1-6-23.xlsx", target_func)
    #pre1.setColumns(icols=range(2,5),pcols=None,mcols=mcols) 
    #pre1.setLimits(lo, hi)
    #X1,y1,c_names1,sblk_names1,indices1 = pre1.prepareData()

        
    #pre2 = pp.PreProcess("../../Data/data-mca-pmu-v1-6-23.xlsx", target_func)
    #pre2 = pp.PreProcess("./data/TimedLBR_SuperBlocks-pmu-mca-v2-winnow.xlsx", target_func)
    #pre2.setColumns(icols=range(2,5),pcols=pcols,mcols=None)
    #pre2.setLimits(lo, hi)
    #X2,y2,c_names2,sblk_names2,indices2 = pre2.prepareData()
    #classes = pre2.createClassificationData(85)
    
    
    #pre3 = pp.PreProcess("../../Data/data-mca-pmu-v1-6-23.xlsx", target_func)
    #pre3.setColumns(icols=range(2,5),pcols=pcols,mcols=mcols)
    #pre3.setLimits(lo, hi)
    #X3,y3,c_names3,sblk_names3,indices3 = pre3.prepareData()
    
    #pre.draw_histogram(30)
    
    #logTrans = False
    
    #sp1 =sup.Supervised(X1,y1,c_names1,logTrans = logTrans)
    #sp2 =sup.Supervised(X2,y2,c_names2,logTrans = logTrans,cls=classes)
    #sp3 =sup.Supervised(X3,y3,c_names3,logTrans = logTrans)
    
    #sp.correlationAnalysis(k=0)
    #sp.plotCorrelations()
    #sp.linearRegressionsModels()
    #sp.plot_scatter()

    #print()
    #print("MCA Only")
    #sp1.randomForestRegressionModels(500,6)
    #print()
    #print("PMU Only")
    #sp2.classification_full_data()
    #print()
    #print("MCA + PMU")
    #sp3.randomForestRegressionModels(500,6)    
    #sp.plot_scatter()
    
    #print(c_names2[0],c_names2[2],c_names2[3])
    
    
    


