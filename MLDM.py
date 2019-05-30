# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

import os
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression  
import sys

#%%
def main(TestBatch, DocumentName):
    Dataset = 'ENRON' #'TREC'
    catR = 'Fraud' #'spam'
    catNR = 'Legit' #'ham' 
    
    start_time = time.time()
    Indexlist = pd.read_csv('Index.csv', sep=";", header=None)
    Indexlist.columns = ['Index', 'document']
    
    #%%
    NrMails = 1640 #75000/5
    SavePer = 40 #100
    
    Traininglist = pd.read_csv('Traininglist'+str(Dataset),sep = '\t', index_col = 0)
    wordselection =pd.read_csv('wordselection-'+str(Dataset)+str(DocumentName), sep = '\t', index_col = 0, names=['Words'])
     
    #%%
    # =============================================================================
    # start of training LOGISTIC REGRESSION
    # =============================================================================
    Training = pd.DataFrame(0, columns = wordselection, index = [], dtype = 'uint32')
    y = pd.DataFrame(0, columns = [], index = [], dtype = 'uint32')
    Batches = list(Traininglist.columns.values)
    Batches.remove(TestBatch)
    
    for batch in Batches:
        print(batch)
        for Files in range(int(SavePer),int(NrMails+SavePer),int(SavePer)):
            TrainingFile = pd.DataFrame() 
            TrainingFile = pd.read_csv('Frequencies'+str(Dataset)+batch+'-'+str(Files),sep = '\t', index_col = 0)
            y = pd.concat([y, TrainingFile['Index_given']], sort=False, ignore_index=True)
            Training = pd.concat([Training, TrainingFile], sort=False, ignore_index=True)
            Training = Training[wordselection]
            Training = Training.fillna(0).to_sparse(fill_value=0)
            print(round(Files/float(NrMails)*100,4), '%')
    
    del TrainingFile
    
    Training.to_csv('Training'+str(Dataset)+'-MLDM',sep='\t')
    
    train = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(Training, y)

    #%%
    # =============================================================================
    # Applying LOGISTIC REGRESSION to test data
    # =============================================================================
    start_time2 = time.time()
    TrainingWords = list(Training.columns.values)
    Test = pd.DataFrame(0, columns = TrainingWords, index = [], dtype = 'uint32')
    
    Traininglist = pd.read_csv('Traininglist'+str(Dataset),sep = '\t', index_col = 0)
    Batches = list(Traininglist.columns.values)
    Batches.remove(TestBatch)
    
    ProbSpam = list()
    ProbHam = list()
    Given_y = list()
    Predicted_y = list()
    for Files in range(int(SavePer),int(NrMails+SavePer),int(SavePer)):
        Test = pd.DataFrame(0, columns = TrainingWords, index = [], dtype = 'uint32')
        TestFile = pd.read_csv('Frequencies'+str(Dataset)+batch+'-'+str(Files),sep = '\t', index_col=0)     
        y = TestFile['Index_given']
        del TestFile['Index_given']
        
        Test = Test.merge(TestFile, how='outer')
        for word in list(set(Test.columns.values)-set(TrainingWords)):
            del Test[word]
                    
        TestNew = Test.fillna(0).to_sparse(fill_value=0)
          
        pred = train.predict(TestNew)
        proba = train.predict_proba(TestNew)
    
        for i in range(0,len(Test)):
            ProbSpam.append(proba[i][0])
            ProbHam.append(proba[i][1])
            Given_y.append(y[i])
            Predicted_y.append(pred[i])
        
        print(round(Files/float(NrMails)*100,4), '%')
        
    Result = pd.DataFrame(0, columns = ["Given_Label", "Predicted_Label", "ProbSpam", "ProbHam", "expSpam", "expHam"], index = [], dtype = 'uint32')
    Result["Given_Label"] =  Given_y
    Result["Predicted_Label"] = Predicted_y
    Result["ProbSpam"] = ProbSpam
    Result["ProbHam"] = ProbHam   
    Result["expSpam"] = 0
    Result["expHam"] = 0  
        
    Result.to_csv('Result'+str(Dataset)+'-'+str(DocumentName),sep='\t')
    
    Timings = pd.DataFrame(columns = ['Description','time'])
    Timings = Timings.append(pd.Series({'Description':'Script', 'time': time.time()-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Training', 'time': start_time2-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Classification', 'time': time.time()- start_time2}), ignore_index=True)
    Timings.to_csv('Timings'+str(Dataset)+'-'+str(DocumentName),sep='\t')


l = ['TL1', 'TL2', 'TL3', 'TL4', 'TL5']
for batch in l:
    main(str(batch), 'MLDM')     