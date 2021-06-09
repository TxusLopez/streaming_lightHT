
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
from river.utils import dict2numpy
#np.random.seed(723)
#init_notebook_mode(connected=True)
#cf.go_offline()

#==============================================================================
# CLASSES
#==============================================================================



#==============================================================================
# FUNCTIONS
#==============================================================================


def data_preparation(datsets,iddata,seed_env,length_dats,drift_width):
    
    ###################### REALS
    
    # if datsets[iddata]=='http':            
    #     dat=dats.HTTP().take(length_dats)               
    # if datsets[iddata]=='credit_card':            
    #     dat=dats.CreditCard().take(length_dats)               
    # if datsets[iddata]=='higgs':            
    #     dat=dats.Higgs().take(length_dats)               
    if datsets[iddata]=='image_segments':            
        dat=dats.ImageSegments().take(length_dats)               
    # elif datsets[iddata]=='insects':            
    #     dat=dats.Insects().take(length_dats)               
    # elif datsets[iddata]=='malicious_url':            
    #     dat=dats.MaliciousURL().take(length_dats)               
    # elif datsets[iddata]=='movielens100K':            
    #     dat=dats.MovieLens100K().take(length_dats)               
    # elif datsets[iddata]=='music':            
    #     dat=dats.Music().take(length_dats)               
    elif datsets[iddata]=='phising':            
        dat=dats.Phishing().take(length_dats)               
    # elif datsets[iddata]=='smtp':            
    #     dat=dats.SMTP().take(length_dats)               
    # elif datsets[iddata]=='trec07':            
    #     dat=dats.TREC07().take(length_dats)               




    ###################### SYNTHS
    
    #AGRAWAL
    elif datsets[iddata]=='agrawal_0_1':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=0),drift_stream=synth.Agrawal(seed=seed_env,classification_function=1),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_1_2':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=1),drift_stream=synth.Agrawal(seed=seed_env,classification_function=2),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_2_3':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=2),drift_stream=synth.Agrawal(seed=seed_env,classification_function=3),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_3_4':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=3),drift_stream=synth.Agrawal(seed=seed_env,classification_function=4),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_4_5':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=4),drift_stream=synth.Agrawal(seed=seed_env,classification_function=5),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_5_6':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=5),drift_stream=synth.Agrawal(seed=seed_env,classification_function=6),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_6_7':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=6),drift_stream=synth.Agrawal(seed=seed_env,classification_function=7),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_7_8':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=7),drift_stream=synth.Agrawal(seed=seed_env,classification_function=8),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='agrawal_8_9':
        dat = synth.ConceptDriftStream(stream=synth.Agrawal(seed=seed_env,classification_function=8),drift_stream=synth.Agrawal(seed=seed_env,classification_function=9),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               

    #MIXED
    elif datsets[iddata]=='mixed':
        dat = synth.ConceptDriftStream(stream=synth.Mixed(classification_function=0,balance_classes=True,seed=seed_env),drift_stream=synth.Mixed(classification_function=1,balance_classes=True,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               

    #RANDOM_RBF                
    elif datsets[iddata]=='randomRBF':
        dat = synth.ConceptDriftStream(stream=synth.RandomRBFDrift(n_centroids=50,change_speed=0.0,n_drift_centroids=50,seed_sample=seed_env,seed_model=seed_env),drift_stream=synth.RandomRBFDrift(n_centroids=10,change_speed=0.0,n_drift_centroids=10,seed_sample=seed_env,seed_model=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               

    #SEA
    elif datsets[iddata]=='sea_0_1':
        dat = synth.ConceptDriftStream(stream=synth.SEA(variant=0,seed=seed_env),drift_stream=synth.SEA(variant=1,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='sea_1_2':
        dat = synth.ConceptDriftStream(stream=synth.SEA(variant=1,seed=seed_env),drift_stream=synth.SEA(variant=2,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='sea_2_3':
        dat = synth.ConceptDriftStream(stream=synth.SEA(variant=2,seed=seed_env),drift_stream=synth.SEA(variant=3,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               

    #STAGGER
    elif datsets[iddata]=='stagger_0_1':
        dat = synth.ConceptDriftStream(stream=synth.STAGGER(classification_function=0,balance_classes=True,seed=seed_env),drift_stream=synth.STAGGER(classification_function=1,balance_classes=True,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='stagger_1_2':
        dat = synth.ConceptDriftStream(stream=synth.STAGGER(classification_function=1,balance_classes=True,seed=seed_env),drift_stream=synth.STAGGER(classification_function=2,balance_classes=True,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               

    #SINE
    elif datsets[iddata]=='sine_0_1':
        dat = synth.ConceptDriftStream(stream=synth.Sine(classification_function=0,balance_classes=True,seed=seed_env),drift_stream=synth.Sine(classification_function=1,balance_classes=True,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='sine_1_2':
        dat = synth.ConceptDriftStream(stream=synth.Sine(classification_function=1,balance_classes=True,seed=seed_env),drift_stream=synth.Sine(classification_function=2,balance_classes=True,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               
    elif datsets[iddata]=='sine_2_3':
        dat = synth.ConceptDriftStream(stream=synth.Sine(classification_function=2,balance_classes=True,seed=seed_env),drift_stream=synth.Sine(classification_function=3,balance_classes=True,seed=seed_env),seed=seed_env, position=int(length_dats/2), width=drift_width).take(length_dats)               

    return dat
    
#######################################################################
########################### MAIN 
#######################################################################
# Ignore warnings
warnings.simplefilter("ignore")


#==============================================================================
# VARIABLES
#==============================================================================

############# PARAMETERS CONFIGURATION ###################################
        
length_dataset=10000
change_width=1
seed=48

path_data='D:\\OPTIMA\\Publicaciones\\ICDM_2021\\DATA\\data_csv\\'

datas=['agrawal_0_1','agrawal_1_2','agrawal_2_3','agrawal_3_4','agrawal_4_5','agrawal_5_6','agrawal_6_7',
        'agrawal_7_8','agrawal_8_9','mixed','randomRBF','sea_0_1','sea_1_2','sea_2_3','stagger_0_1','stagger_1_2',
        'sine_0_1','sine_1_2','sine_2_3','image_segments','phising']#http,credit_card,higgs,insects,malicious_url,movielens100K,music,smtp,trec07
    
try:
                    
    for d in range(len(datas)):        

        #######################################################
        #### DATA PREPARATION
        
        data=data_preparation(datas,d,seed,length_dataset,change_width)                                           
        features_df=pd.DataFrame()
        labels=[]
        
        for x, y in data:
            s=dict2numpy(x)
            s2=pd.DataFrame(s)
            df=s2.T        
            
            columns=list(x.keys())            
            
            features_df=features_df.append(df)
            labels.append(y)

        # features_df.index=np.arange(0, length_dataset)
        features_df.columns=columns
        features_df['label']=labels
        df_data=features_df

        #######################################################
        #### DATA STORAGE
        df_data.to_csv(path_data+datas[d]+'.csv',sep=',',header=0,index=False)


except Exception as e:
    print('Excepcion: ',e)
    traceback.print_exc()   
