# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:37:56 2019

@author: dkaak
https://www.kaggle.com/gkoundry/bayesian-logistic-regression-with-pystan
"""

import os
import numpy as np
import pystan
import time
import pandas as pd

# Create model code
line_code = """
data {
    int N;
    int N2;
    int D;
    int y[N];
    matrix[N, D] x;
    matrix[N2, D] x2;
}
parameters {
    vector[D] beta;
    real alpha;
}
transformed parameters {
        vector[N] linpred;
        linpred = alpha+x*beta;
}
model {
    alpha ~ normal(0,10^6);
    for (i in 1:D)
        beta[i] ~ normal(0,10^6);
    y ~ bernoulli_logit(linpred);
}
generated quantities {                                                                               
  vector[N2] ypred;                                                                            
  ypred = alpha+x2*beta;                               
}   
"""

def main(Testbatch):
    start_time = time.time()

    Dataset = 'ENRON' #'TREC'
    catR = 'Fraud' #'spam'
    catNR = 'Legit' #'ham' 
    
    DocumentName = 'BEDM-'+str(Dataset)+'-'+str(Testbatch)
    
    start_time = time.time()
    Indexlist = pd.read_csv('Index.csv', sep=";", header=None)
    Indexlist.columns = ['Index', 'document']
    
    NrMails = 1640 #75000/5
    SavePer = 40 #100
    
    Traininglist = pd.read_csv('Traininglist'+str(Dataset),sep = '\t', index_col = 0)
    wordselection =pd.read_csv('wordselection-'+str(Dataset)+str(DocumentName), sep = '\t', index_col = 0, names=['Words'])

    start_time2 = time.time()
    
    Training = pd.DataFrame(0, columns = wordselection, index = [], dtype = 'uint32')
    y = pd.DataFrame(0, columns = [], index = [], dtype = 'uint32')
    Batches = list(Traininglist.columns.values)
    Batches.remove(Testbatch)
    
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
    
    y = y.replace(str(catR),1)
    y = y.replace(str(catNR),0)
    
    start_time3 = time.time()
    
    batch = Testbatch
    Test = pd.DataFrame(0, columns = wordselection, index = [], dtype = 'uint32')
    y2 = pd.DataFrame(0, columns = [], index = [], dtype = 'uint32')
    for Files in range(int(SavePer),int(NrMails+SavePer),int(SavePer)):
        TestFile = pd.DataFrame() 
        TestFile = pd.read_csv('Frequencies'+str(Dataset)+batch+'-'+str(Files),sep = '\t', index_col = 0)
        y2 = pd.concat([y2, TestFile['Index_given']], sort=False, ignore_index=True)
        Test = pd.concat([Test, TestFile], sort=False, ignore_index=True)
        Test = Test[wordselection]
        Test = Test.fillna(0).to_sparse(fill_value=0)
        print(round(Files/float(NrMails)*100,4), '%')
    
    y2 = y2.replace(str(catR),1)
    y2 = y2.replace(str(catNR),0)
    
    start_time4 = time.time()
    linear_data = {'N': Training.shape[0],
                   'N2': Test.shape[0],
                   'D': len(wordselection),
                   'y': np.array(y[0]),       
                   'x': np.array(Training),
                   'x2':np.array(Test)} 
    
    Nsamples = 1000
    chains = 1  
    
    sm = pystan.StanModel(model_code=line_code); 
    fit = sm.sampling(data=linear_data, iter=Nsamples, chains=chains,algorithm="NUTS", n_jobs=-1);
    a = fit.extract(permuted=False)
    b = fit.extract(pars='ypred')
    
    start_time5 = time.time()
    
    Result = pd.DataFrame(columns = ["Predicted_Label", "Given_Label"])
    for j in range(0,b['ypred'].shape[1]):
        c = 0
        for i in range(0,int(Nsamples/2)):
            c = c + b['ypred'][i][j]
        c = c/(Nsamples/2)
        if c > 0:
            Result = Result.append({'Predicted_Label': 1, "Given_Label":y2[0][j]}, ignore_index=True)
        else:
            Result = Result.append({'Predicted_Label': 0, "Given_Label":y2[0][j]}, ignore_index=True)
    
    Result.to_csv('Result-'+str(DocumentName),sep='\t')
    
    Timings = pd.DataFrame(columns = ['Description','time'])
    Timings = Timings.append(pd.Series({'Description':'Script', 'time': time.time()-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Intro', 'time': start_time2-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'TrainingFile', 'time': start_time3-start_time2}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'TestFile', 'time': start_time4-start_time3}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'StanModel', 'time': start_time5- start_time4}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Test', 'time': time.time()- start_time5}), ignore_index=True)
    Timings.to_csv('Timings-'+str(DocumentName)+'Test',sep='\t')
    
l = ['TL1', 'TL2', 'TL3', 'TL4', 'TL5']
for batch in l:
    main(batch)