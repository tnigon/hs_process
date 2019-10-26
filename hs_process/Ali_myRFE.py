# -*- coding: utf-8 -*-
"""
Feature Selection by SVM-RFE

"""
from sklearn.feature_selection import RFE
#from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

#import pickle
#out_file = r'C:\Users\moghi005\Documents\MATLAB\Feature Selection\DataSet_Feature Selection\seclectorRFECV.obj' # .obj file


def myRFE(X, y):

    # classification algorithm
    # To use RFE, it is a must to have a supervised learning estimator which attribute coef_ is available,
    # this is the case only for the linear kernel. 
    # coef_ is not the case for SVM using other kernels different from Linear.
    estimator = SVR(kernel="linear")
    
    # Recursive Feature Elimination - returning feature importance 
    #myrfecv = myRFE_weight(estimator, step=214)
    #selectorRFE = myrfecv.fit(X, y)
    
    # Recursive Feature Elimination - Cross Validation
    '''
    NOTE:
    RFECV performs RFECV in a cross-validation loop to find the optimal number of features
    Since we want to the algorithm remove all the features except the most important one, then we can use RFE and not RFECV
    '''
    #rfecv = RFECV(estimator, step=1, cv=5)
    #selectorRFECV = rfecv.fit(X, y)
    #selectedFt = selectorRFECV.support_ 
    #featureRanking = selectorRFECV.ranking_
    
    # Recursive Feature Elimination 
    rfe = RFE(estimator, 1, step=1)
    selectorREF = rfe.fit(X, y)
    featureRanking = selectorREF.ranking_
    
    return featureRanking
    
    
    
    #print('The scikit-learn version is {}.'.format(sklearn.__version__))
    #
    #
    ## save the result as an object 
    #file_rfe = open('filename_pi.obj', 'wb') 
    #pickle.dump(featureRanking, file_rfe) 
    #file_rfe.close()
    
    # loading the save object back 
    #filehandler = open('filename_pi.obj', 'rb') 
    #object_pi = pickle.load(filehandler) 
    #print ('loaded_obj is', object_pi)
    #
    #ftRanking = object_pi.ranking_