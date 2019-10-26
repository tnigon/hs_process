# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:42:17 2017

@author: moghi005
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#import math


def myRandomForest(X, y, wavelength):
    
    n_samples, n_features = X.shape
    
    
    RandForest = RandomForestClassifier(n_estimators=1000,
                                         criterion='entropy',
                                         max_depth=None, # depth of the tree - number of levels
                                         min_samples_split=10,
                                         min_samples_leaf=10,
                                         min_weight_fraction_leaf=0.0,
                                         max_features='auto',
                                         max_leaf_nodes=None,
                                         bootstrap=True,
                                         oob_score=False,
                                         n_jobs=1, random_state=None,
                                         verbose=0, warm_start=False,
                                         class_weight=None)
    
    RandForest.fit(X, y)
    
    # ============== membership probability of each samples in each class ======== 
    #result = RandForest.predict_proba(x_test)
    ## assign a class to samples according to the probability of their membership
    #y_predicted = np.empty(len(y_test), dtype=object)
    #
    #for elem in range (len(y_test)):
    #            compare_prob = result[elem,1] > result[elem,0]
    #            assigned_class = int(compare_prob)
    #            y_predicted[elem] = assigned_class
    
    # ============= Binary Classification ========================================
    #y_predicted = RandForest.predict(x_test2)
    
    
    # features importance
    importances = RandForest.feature_importances_ 
    feature_ranking = np.argsort(importances)[::-1] + 1 # 
    
    # In[]
    
#    std = np.std([tree.feature_importances_ for tree in RandForest.estimators_], axis=0)
    #indices = np.argsort(importances)[::-1] + 1
    
    # Print the feature ranking
    #print("Feature ranking:")
    #
    #for f in range(x_train.shape[1]):
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    ## Plot the feature importances of the forest
    #ax = plt.figure(figsize=(7, 3))
    ##plt.title("Feature importances")
    ##plt.bar(range(x.shape[1]), importances[indices],
    ##       color="r", yerr=std[indices], align="center")
    #plt.bar(range(x_train.shape[1]), importances[indices], color="r", align = "center", width=0.6)
    #
    #plt.xticks(range(x_train.shape[1]), [math.trunc(elem) for elem in wavelength[indices]], fontsize=6, rotation=90)
    #plt.yticks(fontsize=7)
    #plt.tick_params(axis='x', which='major', pad=5)
    #plt.xlabel('features')
    #plt.ylabel('feature importance')
    #plt.savefig('feature_importance.png', dpi = 600)
    #plt.show()
    
    # In[]:
#    ax = plt.figure(figsize=(7, 3))
    #plt.title("Feature importances")
    #plt.bar(range(x.shape[1]), importances[indices],
    #       color="r", yerr=std[indices], align="center")
    plt.bar(wavelength, importances, color="r", align = "center", width=0.6)
    
    #plt.xticks([math.trunc(elem) for elem in wavelength], fontsize=10, rotation=90)
    #plt.xticks(np.arange(min(wavelength), max(wavelength)+1, 20), fontsize=10, rotation=90)
    #min_x_axis = np.min(wavelength)
    #max_x_axis = np.max(wavelength)
    
    plt.xticks(np.arange(450, 900, 25), fontsize=16, rotation=90)
    
    plt.yticks(fontsize=14)
    plt.tick_params(axis='x', which='major', pad=5)
    plt.xlabel('wavelength (nm)', fontsize=20)
    plt.ylabel('feature importance', fontsize=20)
    #plt.savefig('feature_importance.png', dpi = 600)
    plt.show()
    
    
    # In[]
        
    #difference = [int(elem) for elem in (y_predicted != y_test)]
    #n_error = sum(difference)
    #ErrorRateTemp = n_error/len(y_test)
    #print(ErrorRateTemp)
    
    
    # In[]:
    # Compute confusion matrix
    #class_names = ['Stress','Control']
    #
    #cnf_matrix = confusion_matrix(y_test, y_predicted)
    #np.set_printoptions(precision=2)
    #
    ## Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                      title='Confusion matrix, without normalization')
    #
    ## Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                      title='Normalized confusion matrix')
    #
    #plt.show()
    
    # In[]:
    
    #print(classification_report(y_test, y_predicted))

    return feature_ranking, importances


