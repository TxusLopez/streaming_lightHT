

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
# import tracemalloc
import random
#import more_itertools as mit
import multiprocessing as mp
import pickle
import scipy.io as sio
import traceback
import creme
import psutil
import itertools

#import seaborn as sns
#sns.set()

#import cufflinks as cf
#import chart_studio.plotly as py
#import plotly.graph_objs as go

from time import time
from matplotlib import pyplot
from copy import deepcopy
from collections import deque
from skmultiflow.drift_detection.page_hinkley import PageHinkley
from sklearn import preprocessing
from scipy.stats import uniform,expon
from collections import deque
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,HalvingRandomSearchCV
from sklearn.metrics import accuracy_score,classification_report,f1_score,confusion_matrix,balanced_accuracy_score
from sklearn.base import clone
from sklearn.metrics import make_scorer,zero_one_loss
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
#from skopt.space import Real, Integer,Categorical
#from skopt.utils import use_named_args
#from skopt import gp_minimize
from skmultiflow.trees import HoeffdingTree
from skmultiflow.lazy import KNN
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.rules import VeryFastDecisionRulesClassifier
#from skopt import BayesSearchCV
from texttable import Texttable
from scipy.spatial.distance import hamming
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from timeit import default_timer as timer
from sklearn.model_selection import cross_validate
from scipy.spatial import distance
#from imblearn.over_sampling import SMOTE,SVMSMOTE,BorderlineSMOTE,SMOTENC,ADASYN
from sklearn.neighbors import NearestNeighbors
from random import randrange, choice
from scipy.spatial import distance_matrix
#from skopt import BayesSearchCV
from skopt.space import Integer,Real,Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE,KMeansSMOTE,SVMSMOTE
from sklearn.svm import SVC
from sklearn.cluster import KMeans,MiniBatchKMeans
from KDEpy.FFTKDE import FFTKDE
from KDEpy.NaiveKDE import NaiveKDE
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
from timeit import default_timer as timer
from river import stream
from sklearn.feature_extraction import DictVectorizer
from collections import deque
from sklearn.model_selection import ParameterSampler
from river import datasets as dats
from river import tree,compat,utils,optim,expert,metrics,evaluate,datasets
from river.drift import ADWIN,PageHinkley
from river import synth
from river import stream
#np.random.seed(723)
#init_notebook_mode(connected=True)
#cf.go_offline()

#==============================================================================
# CLASSES
#==============================================================================



#==============================================================================
# FUNCTIONS
#==============================================================================

def seeking_neighs_candidates(gr,best_est,or_tr_X,or_tr_Y,X_dts,y_dts,it):
        
     # print('it: ',it)
     if it==0:
         # print('devolucion=',best_est.model_description)

         return best_est
         
     else:

        #We select the best, and move 1 step over the parameter values
        pos_grace_val=list(gr['grace_period']).index(best_est.grace_period)
        pos_tie_val=list(gr['tie_threshold']).index(best_est.tie_threshold)
        pos_splitconf_val=list(gr['split_confidence']).index(best_est.split_confidence)
        pos_depth_val=list(gr['max_depth']).index(best_est.max_depth)
        pos_nb_val=list(gr['nb_threshold']).index(best_est.nb_threshold)
        
        #GRACE PERIOD
        if pos_grace_val+1<len(gr['grace_period']):
            grace_val_1=gr['grace_period'][pos_grace_val+1]
        else:
            grace_val_1=gr['grace_period'][pos_grace_val]
            
        if pos_grace_val-1>=0:
            grace_val_2=gr['grace_period'][pos_grace_val-1]
        else:
            grace_val_2=gr['grace_period'][pos_grace_val]
        
        graces=[grace_val_2,grace_val_1]
            
        #TIE THRESHOLD
        if pos_tie_val+1<len(gr['tie_threshold']):
            tie_val_1=gr['tie_threshold'][pos_tie_val+1]
        else:
            tie_val_1=gr['tie_threshold'][pos_tie_val]
            
        if pos_tie_val-1>=0:
            tie_val_2=gr['tie_threshold'][pos_tie_val-1]
        else:
            tie_val_2=gr['tie_threshold'][pos_tie_val]
    
        ties=[tie_val_2,tie_val_1]
    
        #SPLIT CONFIDENCE
        if pos_splitconf_val+1<len(gr['split_confidence']):
            splitconf_val_1=gr['split_confidence'][pos_splitconf_val+1]
        else:
            splitconf_val_1=gr['split_confidence'][pos_splitconf_val]
            
        if pos_splitconf_val-1>=0:
            splitconf_val_2=gr['split_confidence'][pos_splitconf_val-1]
        else:
            splitconf_val_2=gr['split_confidence'][pos_splitconf_val]
        
        splits=[splitconf_val_2,splitconf_val_1]
        
        #MAX DEPTH
        if pos_depth_val+1<len(gr['max_depth']):
            depth_val_1=grid['max_depth'][pos_depth_val+1]
        else:
            depth_val_1=gr['max_depth'][pos_depth_val]
            
        if pos_depth_val-1>=0:
            depth_val_2=gr['max_depth'][pos_depth_val-1]
        else:
            depth_val_2=gr['max_depth'][pos_depth_val]
        
        depths=[depth_val_2,depth_val_1]    
        
        #NB THRESHOLD
        if pos_nb_val+1<len(gr['nb_threshold']):
            nb_val_1=grid['nb_threshold'][pos_nb_val+1]
        else:
            nb_val_1=gr['nb_threshold'][pos_nb_val]
            
        if pos_nb_val-1>=0:
            nb_val_2=gr['nb_threshold'][pos_nb_val-1]
        else:
            nb_val_2=gr['nb_threshold'][pos_nb_val]
        
        nbs=[nb_val_2,nb_val_1]    
        
        # print('grace value: ',gr['grace_period'][pos_grace_val])
        # print('graces: ',graces)

        # print('depths value: ',gr['max_depth'][pos_depth_val])
        # print('depths: ',depths)

        # print('split_confidence value: ',gr['split_confidence'][pos_splitconf_val])
        # print('splits: ',splits)

        # print('tie_threshold value: ',gr['tie_threshold'][pos_tie_val])
        # print('ties: ',ties)

        # print('nb_threshold value: ',gr['nb_threshold'][pos_nb_val])
        # print('nbs: ',nbs)


        perms=list(itertools.product(*[graces,depths,splits,ties,nbs]))
        # print('perms: ',perms[0])
        models=[]
        accs=[]
        
        for a in range(len(perms)):
            
            classifier=best_est.clone()
    
            #Training
            classifier.grace_period=perms[a][0]
            classifier.max_depth=perms[a][1]
            classifier.split_confidence=perms[a][2]
            classifier.tie_threshold=perms[a][3]
            classifier.nb_threshold=perms[a][4]
                    
            for i, row in or_tr_X.iterrows(): 
                lab=list(or_tr_Y.values[i])[0]
                classifier=classifier.learn_one(row,lab)        
                        
            models.append(classifier)
            
            acc_metric=metrics.Accuracy()
            #Testing
            for j, k in X_dts.iterrows():  
                pred_conf=classifier.predict_one(k) 
                lab=y_dts.iloc[j].values[0]
                acc_metric.update(lab, pred_conf)        
                
            accs.append(acc_metric.get())
                
        index_max = np.argmax(accs)
        best=models[index_max]     
        
        # print('END best.model_description: ',best.model_description)        
        
        # print('Max accuracy NEIGHS=',np.max(accs))
        # print('model=',best.model_description)
        
        
        return seeking_neighs_candidates(gr,best,or_tr_X,or_tr_Y,X_dts,y_dts,it-1)

    # return best,np.max(accs)

def checking_direction(grid_params,parameter,bst_estim,best_val,pos_value,X_training,y_training,X_test,y_test,rig,lef):
    
    # old_parameter_val=best_val
    parameter_val=0
    grid_pram=grid_params[parameter]
    # new_param_val=0
    
    if rig:
        
        if pos_value+1<len(grid_pram):
            parameter_val=grid_pram[pos_value+1]
        else:
            parameter_val=grid_pram[pos_value]

    elif lef:
            
        if pos_value-1>=0:
            parameter_val=grid_pram[pos_value-1]
        else:
            parameter_val=grid_pram[pos_value]
    
    #CHECKING IF BETTER THAN PREVIOUS MODEL
    '''
    cl=bst_estim.clone()
    
    cl.grace_period=parameter_val
    cl.max_depth=bst_estim.max_depth
    cl.split_confidence=bst_estim.split_confidence
    cl.tie_threshold=bst_estim.tie_threshold
    cl.nb_threshold=bst_estim.nb_threshold
    
    for i, row in X_training.iterrows():  
        lab=list(y_training.values[i])[0]
        cl=classifier.learn_one(row,lab)        
                        
    acc_metr=metrics.Accuracy()
    #Testing
    for j, k in X_test.iterrows():  
        pred_conf=cl.predict_one(k) 
        lab=y_test.iloc[j].values[0]
        acc_metr.update(lab, pred_conf)        
        
    if acc_metr.get()>best_val:
        new_param_val=parameter_val
    else:
        new_param_val=old_parameter_val    
    '''
    
    return parameter_val

def seeking_direct_candidates(gr,best_est,or_tr_X,or_tr_Y,X_dts,y_dts,it,best_acc):

     if it==0:
         # print('devolucion=',best_est.model_description)

         return best_est
         
     else:
        #We select the best, and move towards one direction in each parameter
        pos_grace_val=list(gr['grace_period']).index(best_est.grace_period)
        pos_tie_val=list(gr['tie_threshold']).index(best_est.tie_threshold)
        pos_splitconf_val=list(gr['split_confidence']).index(best_est.split_confidence)
        pos_depth_val=list(gr['max_depth']).index(best_est.max_depth)
        pos_nb_val=list(gr['nb_threshold']).index(best_est.nb_threshold)
        
        # print('pos_grace_val: ',pos_grace_val)
        # print('pos_tie_val: ',pos_tie_val)
        # print('pos_splitconf_val: ',pos_splitconf_val)
        # print('pos_depth_val: ',pos_depth_val)
        # print('pos_nb_val: ',pos_nb_val)
        
        right_grace=False
        left_grace=False
        if random.uniform(0, 1)<0.5:
            left_grace=True
        else:
            right_grace=True

        right_tie=False
        left_tie=False
        if random.uniform(0, 1)<0.5:
            left_tie=True
        else:
            right_tie=True

        right_splitconf=False
        left_splitconf=False
        if random.uniform(0, 1)<0.5:
            left_splitconf=True
        else:
            right_splitconf=True

        right_depth=False
        left_depth=False
        if random.uniform(0, 1)<0.5:
            left_depth=True
        else:
            right_depth=True

        right_nb=False
        left_nb=False
        if random.uniform(0, 1)<0.5:
            left_nb=True
        else:
            right_nb=True
        
        # print('right: ',right,' - left: ',left)
        new_grace_val=checking_direction(gr,'grace_period',best_est,best_est.grace_period,pos_grace_val,or_tr_X,or_tr_Y,X_dts,y_dts,right_grace,left_grace)
        new_tie_val=checking_direction(gr,'tie_threshold',best_est,best_est.tie_threshold,pos_tie_val,or_tr_X,or_tr_Y,X_dts,y_dts,right_tie,left_tie)
        new_splitconf_val=checking_direction(gr,'split_confidence',best_est,best_est.split_confidence,pos_splitconf_val,or_tr_X,or_tr_Y,X_dts,y_dts,right_splitconf,left_splitconf)
        new_depth_val=checking_direction(gr,'max_depth',best_est,best_est.max_depth,pos_depth_val,or_tr_X,or_tr_Y,X_dts,y_dts,right_depth,left_depth)
        new_nb_val=checking_direction(gr,'nb_threshold',best_est,best_est.nb_threshold,pos_nb_val,or_tr_X,or_tr_Y,X_dts,y_dts,right_nb,left_nb)

        # print('new_grace_val: ',new_grace_val)
        # print('new_tie_val: ',new_tie_val)
        # print('new_splitconf_val: ',new_splitconf_val)
        # print('new_depth_val: ',new_depth_val)
        # print('new_nb_val: ',new_nb_val)

            
        cl=best_est.clone()
        
        cl.grace_period=new_grace_val
        cl.max_depth=new_depth_val
        cl.split_confidence=new_splitconf_val
        cl.tie_threshold=new_tie_val
        cl.nb_threshold=new_nb_val
        
        #Get the accuracy of the new configuration
        for i, row in or_tr_X.iterrows():  
            lab=list(or_tr_Y.values[i])[0]
            cl=cl.learn_one(row,lab)        
                            
        ac=metrics.Accuracy()
        #Testing
        for j, k in X_dts.iterrows():  
            pred_conf=cl.predict_one(k) 
            lab=y_dts.iloc[j].values[0]
            ac.update(lab, pred_conf)                         

        # print('Old ac: ',best_acc)
        # print('Old model: ',best_est.model_description)

        # print('New ac: ',ac.get())
        # print('New model: ',cl.model_description)

                    
        current_ac=best_acc
        current_model=best_est.clone()
        if ac.get()>best_acc:
            current_ac=ac.get()
            current_model=cl.clone()

        # print('Current model: ',current_model.model_description)
        
        
        return seeking_direct_candidates(gr,current_model,or_tr_X,or_tr_Y,X_dts,y_dts,it-1,current_ac)

def tuning_neighs(scor,orig_tr_X,orig_tr_Y,X_dats,y_dats,iterats,params_val,classif):#n_select

    print('tuning_neighs')
    
    selected=seeking_neighs_candidates(params_val,classif,orig_tr_X,orig_tr_Y,X_dats,y_dats,iterats)#best_estimator
    
    def_mod=selected.clone()
    #Train the selected
    for i, row in orig_tr_X.iterrows():  
        lab=list(orig_tr_Y.values[i])[0]
        def_mod=def_mod.learn_one(row,lab)            
    
    return def_mod

def tuning_direct(scor,orig_tr_X,orig_tr_Y,X_dats,y_dats,iterats,params_val,classif):#n_select

    print('tuning_direct')

    acc_metric=metrics.Accuracy()
    #Testing
    for j, k in X_dats.iterrows():  
        pred_conf=classif.predict_one(k) 
        lab=y_dats.iloc[j].values[0]
        acc_metric.update(lab, pred_conf)        
        
    selected=seeking_direct_candidates(params_val,classif,orig_tr_X,orig_tr_Y,X_dats,y_dats,iterats,acc_metric.get())#best_estimator
    
    def_mod=selected.clone()
    #Train the selected
    for i, row in orig_tr_X.iterrows():  
        lab=list(orig_tr_Y.values[i])[0]
        def_mod=def_mod.learn_one(row,lab)            
    
    return def_mod
           
def prequential_acc(predicted_class,Y_tst,PREQ_ACCS,t_step,fact):

    #Prequential accuracy
    pred=0
    if predicted_class==Y_tst:    
        pred=1
    else:
        pred=0

    if t_step==0:
        preqAcc=1
    else:   
        preqAcc=(PREQ_ACCS[-1]+float((pred-PREQ_ACCS[-1])/(t_step-fact+1)))

    return preqAcc


def data_preparation(datsets,iddata,data_path):
    
    dat=pd.read_csv(data_path+datsets[iddata]+'.csv',sep=',',header=None)
    length_dats=dat.shape[0]

     #AGRAWAL
    if datsets[iddata]=='agrawal_0_1':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_1_2':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_2_3':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_3_4':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_4_5':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_5_6':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_6_7':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_7_8':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='agrawal_8_9':
        drift_pts=[int(length_dats/2)]
        
    #MIXED
    elif datsets[iddata]=='mixed':
        drift_pts=[int(length_dats/2)]   
             
    #RANDOM_RBF                
    elif datsets[iddata]=='randomRBF':
        drift_pts=[int(length_dats/2)]
        
    #SEA
    elif datsets[iddata]=='sea_0_1':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='sea_1_2':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='sea_2_3':
        drift_pts=[int(length_dats/2)]
        
    #STAGGER
    elif datsets[iddata]=='stagger_0_1':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='stagger_1_2':
        drift_pts=[int(length_dats/2)]
        
    #SINE
    elif datsets[iddata]=='sine_0_1':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='sine_1_2':
        drift_pts=[int(length_dats/2)]
    elif datsets[iddata]=='sine_2_3':
        drift_pts=[int(length_dats/2)]
        
    ###################### REALS
    # elif datsets[iddata]=='http':
    #     drift_pts=[2500,5000,7500]

    # elif datsets[iddata]=='credit_card':
    #     drift_pts=[2500,5000,7500]

    # elif datsets[iddata]=='higgs':
    #     drift_pts=[2500,5000,7500]

    elif datsets[iddata]=='image_segments':
        drift_pts=[500,1000,1500]

    # elif datsets[iddata]=='malicious_url':
    #     drift_pts=[2500,5000,7500]

    # elif datsets[iddata]=='movielens100K':
    #     drift_pts=[2500,5000,7500]

    # elif datsets[iddata]=='music':
    #     drift_pts=[150,300,450]

    elif datsets[iddata]=='phising':
        drift_pts=[300,600,900]

    # elif datsets[iddata]=='smtp':
    #     drift_pts=[2500,5000,7500]
    
    return dat,drift_pts
    
def search_the_best_and_worst(scor,X_tr,y_tr,X_tes,y_tes,combinations):

    models=[]
    accs=[]
    
    for a in range(len(combinations)):
        
        classifier=combinations[a]
        
        for i, row in X_tr.iterrows(): 
            lab=list(y_tr.values[i])[0]
            classifier=classifier.learn_one(row,lab)        
                    
        models.append(classifier)
        
        acc_metric=metrics.Accuracy()
        #Testing
        for j, k in X_tes.iterrows():  
            pred_conf=classifier.predict_one(k) 
            lab=y_tes.iloc[j].values[0]
            acc_metric.update(lab, pred_conf)        
            
        accs.append(acc_metric.get())
            
    return np.max(accs),np.min(accs)

#######################################################################
########################### MAIN 
#######################################################################
# Ignore warnings
warnings.simplefilter("ignore")


#==============================================================================
# VARIABLES
#==============================================================================

############# PARAMETERS CONFIGURATION ###################################
        
grace_period=np.array([25,75,100,130,180,230,290,340,420,500])#[25,130,230,340,500]#[25,75,100,130,180,230,290,340,420,500]
tie_threshold=np.linspace(0.001,1.0,10)#5,10
split_confidence=np.linspace(0.000000001,0.1,10)#5,10
max_depth=np.arange(2, 22,2)#np.arange(2, 22, 4)#2,4
nb_threshold=np.array([25,75,100,130,180,230,290,340,420,500])#[25,130,230,340,500]#[25,75,100,130,180,230,290,340,420,500]

#For RGS and Halving
grid = dict(
    grace_period=grace_period,
    tie_threshold=tie_threshold,
    split_confidence=split_confidence,
    max_depth=max_depth,
    nb_threshold=nb_threshold
)    

n_total_combinations=len(grace_period)*len(tie_threshold)*len(split_confidence)*len(max_depth)*len(nb_threshold)
  


#length_dataset=10000
# change_width=1
scoring='accuracy'#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#cv=10
tst_size=0.25
#n_calls=100#50,100
runs=1#25
#search_space_size=10#10
seed=48
window_size=50#50,100,300

################### NEIGHS AND DIRECT TUNING
#n_initial_neighs_models=50#5,10,50,n_total_combinations
#n_initial_direct_smart_models=2#5,10,50,n_total_combinations
iterations_neighs=3
iterations_direct=3

################### SUCCESIVE_HALVING
eta=2#5,10
max_budget=int((2*n_total_combinations*window_size)/eta)
budget=max_budget#25000,max_budget

n_models_sh_random=50
budget_random=1000

#DRIFT DETECTION
#delt=0.005#0.002,0.005

path_data='D:\\OPTIMA\\Publicaciones\\ICDM_2021\\DATA\\data_csv\\'
path_results='D:\\OPTIMA\\Publicaciones\\ICDM_2021\\RESULTS\\'

datas=['agrawal_0_1','agrawal_1_2','agrawal_2_3','agrawal_3_4','agrawal_4_5','agrawal_5_6','agrawal_6_7',
        'agrawal_7_8','agrawal_8_9','mixed','randomRBF','sea_0_1','sea_1_2','sea_2_3','stagger_0_1','stagger_1_2',
        'sine_0_1','sine_1_2','sine_2_3','image_segments','phising']#http,credit_card,higgs,insects,malicious_url,movielens100K,music,smtp


#==============================================================================
# PROCESSING
#==============================================================================        
    
try:

    OPT_DAT_times_Halving_random,OPT_DAT_times_Halving,OPT_DAT_times_Neighs,OPT_DAT_times_Direct=[],[],[],[]
    OPT_DAT_ramh_Halving_random,OPT_DAT_ramh_Halving,OPT_DAT_ramh_Neighs,OPT_DAT_ramh_Direct=[],[],[],[]
    DAT_preq_acc_Halving_random,DAT_preq_acc_Halving,DAT_preq_acc_Neighs,DAT_preq_acc_Direct=[],[],[],[]
                    
    for d in range(len(datas)):        

        #For OPTIMIZATION results    
        OPT_RUN_times_Halving_random,OPT_RUN_times_Halving,OPT_RUN_times_Neighs,OPT_RUN_times_Direct=[],[],[],[]
        OPT_RUN_ramh_Halving_random,OPT_RUN_ramh_Halving,OPT_RUN_ramh_Neighs,OPT_RUN_ramh_Direct=[],[],[],[]
        #For AFTER DRIFT results    
        RUN_preq_acc_Halving_random,RUN_preq_acc_Halving,RUN_preq_acc_Neighs,RUN_preq_acc_Direct=[],[],[],[]

        #######################################################
        #### DATA PREPARATION
        
        data,drift_points=data_preparation(datas,d,path_data)                                           

        for r in range(runs):

            print('DATASET: ',d)
            print('RUN: ',r)
            
            # rnd_state=r

            preqAccs_SH=[]
            preqAccs_SH_random=[]
            preqAccs_Neighs=[]
            preqAccs_Direc=[]
            
            f_SH=1
            f_SH_random=1
            f_Neighs=1
            f_Direc=1

            times_Halving,times_Halving_random,times_Neighs,times_Direc=[],[],[],[]
            ramh_Halving,ramh_Halving_random,ramh_Neighs,ramh_Direc=[],[],[],[]
            preq_accs_Halving_random,preq_accs_Halving,preq_accs_Neighs,preq_accs_Direc=[],[],[],[]

            #######################################################
            #### INITIALIZATION
                    
            classifier=tree.HoeffdingTreeClassifier()   
                    
            #Assigning initial values
            classifier.grace_period=random.choice(grace_period)
            classifier.max_depth=random.choice(max_depth)
            classifier.tie_threshold=random.choice(tie_threshold)
            classifier.split_confidence=random.choice(split_confidence)
            classifier.nb_threshold=random.choice(nb_threshold)
            
            classifier_Halving=classifier.clone()
            classifier_Halving_random=classifier.clone()
            classifier_Neighs=classifier.clone()
            classifier_NO_OPT=classifier.clone()
            classifier_Direc=classifier.clone()

            #For SH process            
            m=classifier_Halving.clone()                            
            ms = utils.expand_param_grid(m,grid)                                    
            sh = expert.SuccessiveHalvingClassifier(models=ms,metric=metrics.Accuracy(),budget=budget,eta=eta,verbose=True)                                

            #For SH_random process                       
            ms_random = random.sample(ms,n_models_sh_random)                        
            sh_random = expert.SuccessiveHalvingClassifier(models=ms_random,metric=metrics.Accuracy(),budget=budget_random,eta=eta,verbose=True)                                
                        
            initial_samples=[]
            initial_labels=[]
            drift_count=0
            sliding_window_X=deque(maxlen=window_size)
            sliding_window_y=deque(maxlen=window_size)
            
            index=0
                        
            #######################################################
            #### STREAMINGdata
            df_X=data.iloc[:,0:len(data.columns)-1]
            df_y=data.iloc[:,-1]
            
            for x,y in stream.iter_pandas(df_X,df_y):

                sliding_window_X.appendleft(x)
                sliding_window_y.appendleft(y)
                                
                if index<window_size:
                    
                    classifier_NO_OPT=classifier_NO_OPT.learn_one(x, y)
                    classifier_Halving=classifier_Halving.learn_one(x, y)
                    classifier_Halving_random=classifier_Halving_random.learn_one(x, y)
                    classifier_Neighs=classifier_Neighs.learn_one(x, y)
                    classifier_Direc=classifier_Direc.learn_one(x, y)
            
                    preqAccs_SH.append(0)
                    preqAccs_SH_random.append(0)
                    preqAccs_Neighs.append(0)
                    preqAccs_Direc.append(0)
                                
                else:

                    ########### Check best configuarion
                    # if index in list(np.asarray(drift_points)+window_size+1):

                    #     X=pd.DataFrame(sliding_window_X)     
                    #     Y=pd.DataFrame(np.array(sliding_window_y))

                    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=tst_size, random_state=index)

                    #     best,worst=search_the_best_and_worst(scoring,X_train.reset_index(drop=True),y_train.reset_index(drop=True),X_test.reset_index(drop=True),y_test.reset_index(drop=True),ms)
                    #     print('Best Acc: ',best)
                    #     print('Worst Acc: ',worst)
                            
                    ########### Neighbors approach
                    y_pred_Neighs = classifier_Neighs.predict_one(x)

                    preqAccs_Neighs.append(prequential_acc(y_pred_Neighs,y,preqAccs_Neighs,index,f_Neighs))

                    classifier_Neighs = classifier_Neighs.learn_one(x,y)
                        
                    if index in list(np.asarray(drift_points)+window_size+1):
                        # print('Change detected by detector_Neighs in t=',index)
                        
                        neighs_start = timer()
                        neighs_start_ram = psutil.virtual_memory().used#measured in bytes
                                                
                        X=pd.DataFrame(sliding_window_X)     
                        Y=pd.DataFrame(np.array(sliding_window_y))
                        
                        f_Neighs=index
                                                
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=tst_size, random_state=index)
                        # classifier_Neighs=tuning_neighs(scoring,X_train.reset_index(drop=True),y_train.reset_index(drop=True),X_test.reset_index(drop=True),y_test.reset_index(drop=True),mods,iterations,grid,classifier,n_initial_neighs_models)#random_grid         
                        classifier_Neighs=tuning_neighs(scoring,X_train.reset_index(drop=True),y_train.reset_index(drop=True),X_test.reset_index(drop=True),y_test.reset_index(drop=True),iterations_neighs,grid,classifier_Neighs.clone())#random_grid         
                        
                        neighs_time=(timer() - neighs_start)
                        neighs_process_ram=psutil.virtual_memory().used-neighs_start_ram
                        if neighs_process_ram<0:
                            neighs_process_ram=0
                
                        # print('neighs_time: ',neighs_time)
                        times_Neighs.append(neighs_time)
                        ramh_Neighs.append(neighs_process_ram)
                
                    ########### Direct approach
                
                    y_pred_Direc = classifier_Direc.predict_one(x)

                    preqAccs_Direc.append(prequential_acc(y_pred_Direc,y,preqAccs_Direc,index,f_Direc))

                    classifier_Direc = classifier_Direc.learn_one(x,y)
                                        
                    if index in list(np.asarray(drift_points)+window_size+1):
                        # print('Change detected by detector_Direct in t=',index)
                    
                        direct_start = timer()
                        direct_start_ram = psutil.virtual_memory().used#measured in bytes
                        
                        X=pd.DataFrame(sliding_window_X)     
                        Y=pd.DataFrame(np.array(sliding_window_y))
                        
                        f_Direc=index
                                                
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=tst_size, random_state=index)
                        # classifier_Direc=tuning_direct(scoring,X_train.reset_index(drop=True),y_train.reset_index(drop=True),X_test.reset_index(drop=True),y_test.reset_index(drop=True),mods,iterations,grid,classifier,n_initial_direct_smart_models)#random_grid         
                        classifier_Direc=tuning_direct(scoring,X_train.reset_index(drop=True),y_train.reset_index(drop=True),X_test.reset_index(drop=True),y_test.reset_index(drop=True),iterations_direct,grid,classifier_Direc.clone())#random_grid         
                        
                        direct_time=(timer() - neighs_start)
                        direct_process_ram=psutil.virtual_memory().used-direct_start_ram
                        if direct_process_ram<0:
                            direct_process_ram=0
                
                        # print('direct_time: ',direct_time)
                        times_Direc.append(direct_time)
                        ramh_Direc.append(direct_process_ram)
                                            
                            
                    ########### Succesive Halving
                    
                    y_pred_SH = classifier_Halving.predict_one(x)

                    preqAccs_SH.append(prequential_acc(y_pred_SH,y,preqAccs_SH,index,f_SH))

                    classifier_Halving = classifier_Halving.learn_one(x, y)
    
                    if index in list(np.asarray(drift_points)+window_size+1):
                        # print('Change detected by detector_SH in t=',index)
                        
                        sh_start = timer()
                        sh_start_ram = psutil.virtual_memory().used#measured in bytes
                                                                            
                        f_SH=index
                        
                        ev=evaluate.progressive_val_score(dataset=zip(sliding_window_X,sliding_window_y),model=sh,metric=metrics.Accuracy())#,show_time=True,show_memory=True,print_every=10                        
                                                                               
                        classifier_Halving=sh.best_model.clone()
                        
                        for i, row in X.iterrows():  
                            lab=Y.values[i]
                            classifier_Halving=classifier_Halving.learn_one(row,list(lab)[0])     
                            
                        sh_time=(timer() - sh_start)
                        sh_process_ram=psutil.virtual_memory().used-sh_start_ram
                        if sh_process_ram<0:
                            sh_process_ram=0
                            
                        times_Halving.append(sh_time)
                        ramh_Halving.append(sh_process_ram)
                                                                                        
                    ########### Succesive Halving random
                    
                    y_pred_SH_random = classifier_Halving_random.predict_one(x)

                    preqAccs_SH_random.append(prequential_acc(y_pred_SH_random,y,preqAccs_SH_random,index,f_SH_random))

                    classifier_Halving_random = classifier_Halving_random.learn_one(x, y)
    
                    if index in list(np.asarray(drift_points)+window_size+1):
                        # print('Change detected by detector_SH_random in t=',index)
                        
                        sh_random_start = timer()
                        sh_random_start_ram = psutil.virtual_memory().used#measured in bytes
                                                                            
                        f_SH_random=index
                        
                        ev_random=evaluate.progressive_val_score(dataset=zip(sliding_window_X,sliding_window_y),model=sh_random,metric=metrics.Accuracy())#,show_time=True,show_memory=True,print_every=10                        
                                                                               
                        classifier_Halving_random=sh_random.best_model.clone()
                        
                        for i, row in X.iterrows():  
                            lab=Y.values[i]
                            classifier_Halving_random=classifier_Halving_random.learn_one(row,list(lab)[0])     
                            
                        sh_random_time=(timer() - sh_start)
                        sh_random_process_ram=psutil.virtual_memory().used-sh_random_start_ram
                        if sh_random_process_ram<0:
                            sh_random_process_ram=0
                            
                        times_Halving_random.append(sh_random_time)
                        ramh_Halving_random.append(sh_random_process_ram)                    
                
                
                index+=1
                                
                                                        
            OPT_RUN_times_Halving.append(times_Halving)
            OPT_RUN_times_Halving_random.append(times_Halving_random)
            OPT_RUN_times_Neighs.append(times_Neighs)
            OPT_RUN_times_Direct.append(times_Direc)

            OPT_RUN_ramh_Halving.append(ramh_Halving)
            OPT_RUN_ramh_Halving_random.append(ramh_Halving_random)
            OPT_RUN_ramh_Neighs.append(ramh_Neighs)
            OPT_RUN_ramh_Direct.append(ramh_Direc)

            RUN_preq_acc_Halving.append(preqAccs_SH)
            RUN_preq_acc_Halving_random.append(preqAccs_SH_random)
            RUN_preq_acc_Neighs.append(preqAccs_Neighs)
            RUN_preq_acc_Direct.append(preqAccs_Direc)

        '''
        ######################## PLOTTING ########################    
        title=datas[d]
        colors=['b','g','r','y','m','c','pink','k','orange','palegreen','gold'] 

        plotAllinOne(title,RUN_preq_acc_Halving_random,RUN_preq_acc_Halving,RUN_preq_acc_Neighs,RUN_preq_acc_Direct,
                     OPT_RUN_times_Halving_random,OPT_RUN_times_Halving,OPT_RUN_times_Neighs,OPT_RUN_times_Direct,
                     OPT_RUN_ramh_Halving_random,OPT_RUN_ramh_Halving,OPT_RUN_ramh_Neighs,OPT_RUN_ramh_Direct,
                     'Random SH','SH','Neighbors Search','Directions Search',colors,path_results,drift_points,window_size)

        showResults(RUN_preq_acc_Halving_random,RUN_preq_acc_Halving,RUN_preq_acc_Neighs,RUN_preq_acc_Direct,
                     OPT_RUN_times_Halving_random,OPT_RUN_times_Halving,OPT_RUN_times_Neighs,OPT_RUN_times_Direct,
                     OPT_RUN_ramh_Halving_random,OPT_RUN_ramh_Halving,OPT_RUN_ramh_Neighs,OPT_RUN_ramh_Direct)
        '''

        OPT_DAT_times_Halving.append(OPT_RUN_times_Halving)
        OPT_DAT_times_Halving_random.append(OPT_RUN_times_Halving_random)
        OPT_DAT_times_Neighs.append(OPT_RUN_times_Neighs)
        OPT_DAT_times_Direct.append(OPT_RUN_times_Direct)
    
        OPT_DAT_ramh_Halving.append(OPT_RUN_ramh_Halving)
        OPT_DAT_ramh_Halving_random.append(OPT_RUN_ramh_Halving_random)
        OPT_DAT_ramh_Neighs.append(OPT_RUN_ramh_Neighs)
        OPT_DAT_ramh_Direct.append(OPT_RUN_ramh_Direct)
        
        DAT_preq_acc_Halving.append(RUN_preq_acc_Halving)
        DAT_preq_acc_Halving_random.append(RUN_preq_acc_Halving_random)
        DAT_preq_acc_Neighs.append(RUN_preq_acc_Neighs)
        DAT_preq_acc_Direct.append(RUN_preq_acc_Direct)
        
                        
    ######################## SAVING ########################    
    
    output = open(path_results+'OPT_DAT_times_Halving.pkl', 'wb')
    pickle.dump(OPT_DAT_times_Halving, output)
    output.close()
    output = open(path_results+'OPT_DAT_times_Halving_random.pkl', 'wb')
    pickle.dump(OPT_DAT_times_Halving_random, output)
    output.close()
    output = open(path_results+'OPT_DAT_times_Neighs.pkl', 'wb')
    pickle.dump(OPT_DAT_times_Neighs, output)
    output.close()
    output = open(path_results+'OPT_DAT_times_Direct.pkl', 'wb')
    pickle.dump(OPT_DAT_times_Direct, output)
    output.close()

    output = open(path_results+'OPT_DAT_ramh_Halving.pkl', 'wb')
    pickle.dump(OPT_DAT_ramh_Halving, output)
    output.close()
    output = open(path_results+'OPT_DAT_ramh_Halving_random.pkl', 'wb')
    pickle.dump(OPT_DAT_ramh_Halving_random, output)
    output.close()
    output = open(path_results+'OPT_DAT_ramh_Neighs.pkl', 'wb')
    pickle.dump(OPT_DAT_ramh_Neighs, output)
    output.close()
    output = open(path_results+'OPT_DAT_ramh_Direct.pkl', 'wb')
    pickle.dump(OPT_DAT_ramh_Direct, output)
    output.close()

    output = open(path_results+'DAT_preq_acc_Halving.pkl', 'wb')
    pickle.dump(DAT_preq_acc_Halving, output)
    output.close()
    output = open(path_results+'DAT_preq_acc_Halving_random.pkl', 'wb')
    pickle.dump(DAT_preq_acc_Halving_random, output)
    output.close()
    output = open(path_results+'DAT_preq_acc_Neighs.pkl', 'wb')
    pickle.dump(DAT_preq_acc_Neighs, output)
    output.close()
    output = open(path_results+'DAT_preq_acc_Direct.pkl', 'wb')
    pickle.dump(DAT_preq_acc_Direct, output)
    output.close()                


except Exception as e:
    print('Excepcion: ',e)
    traceback.print_exc()   
