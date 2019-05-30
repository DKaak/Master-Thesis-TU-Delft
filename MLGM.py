# -*- coding: utf-8 -*-
"""
python "c:\\users\\dkaak\\OneDrive - KPMG\\Documents\\Thesis\\Code\\Spam\\MLGM.py"
"""

import os
import pandas as pd
import time
import numpy as np
import sys

#%%
def main(TestBatch, DocumentName, p):
    Dataset = 'ENRON' #'TREC'
    catR = 'Fraud' #'spam'
    catNR = 'Legit' #'ham'
    
    p = float(p)
    start_time = time.time()
 
    #%%
    NrMails = 1640 #75000/5
    SavePer = 40 #100

    
    #%%
    Prob = pd.read_csv('Prob'+str(Dataset)+'-'+str(DocumentName), sep='\t', index_col = 0)
    Traininglist = pd.read_csv('Traininglist'+str(Dataset),sep = '\t', index_col = 0)
    
    #%%
    # =============================================================================
    # Applying Naive Bayes to test data
    # =============================================================================
    Result = pd.DataFrame(columns = ['Index', 'Predicted_Label','Given_Label'])
    
    batch = TestBatch
    
    start_time2 = time.time()
    for Files in range(int(SavePer),int(NrMails+SavePer),int(SavePer)):
        Freq2 = pd.DataFrame() 
        Freq2 = pd.read_csv('Frequencies'+str(Dataset)+batch+'-'+str(Files),sep = '\t', index_col=0)
        Freq2 = Freq2.fillna(0)
    
        NrPredictions = len(Freq2)
        Wordlist2 = list(Freq2)
        y = Freq2['Index_given']
        Wordlist2.remove('Index_given')
        del Freq2['Index_given']
        for i in range(0,NrPredictions):
            MessageWords = Freq2.columns[Freq2.iloc[i]>0]
            if len(MessageWords)>0:
                ProbSpam = float(p)
                ProbHam  = float(1-p)
                expSpam = 0
                expHam = 0        
                for word in set(set(Wordlist2).intersection(Prob.index)).intersection(MessageWords):
                    newspam = 1
                    newham = 1
                    expSpam2 = 0
                    expHam2 = 0
                    for z in range(0,int(Freq2[word][i])):
                        newspam = newspam*Prob.loc[word][str(catR)]
                        newham = newham*Prob.loc[word][str(catNR)]
                        expSpam2 = expSpam2 + np.floor(np.log10(np.abs(newspam)))
                        expHam2 = expHam2 + np.floor(np.log10(np.abs(newham)))
                        newspam = newspam*(10**abs(np.floor(np.log10(np.abs(newspam)))))
                        newham = newham*(10**abs(np.floor(np.log10(np.abs(newham)))))                                
                    ProbSpam = ProbSpam*newspam
                    ProbHam  = ProbHam*newham
                    expSpam = expSpam + expSpam2 + np.floor(np.log10(np.abs(ProbSpam)))
                    expHam = expHam + expHam2 + np.floor(np.log10(np.abs(ProbHam)))
                    ProbSpam = ProbSpam*(10**(-abs(np.floor(np.log10(np.abs(ProbSpam))))))
                    ProbHam = ProbHam*(10**(-abs(np.floor(np.log10(np.abs(ProbHam))))))              
                
                number = (ProbHam*(1-p))/(ProbSpam*p)
                exponent = expHam-expSpam
        
                if number <1 and exponent > 100:
                    answer = 10
                elif number > 1 and exponent > 100:
                    answer = 10
                else:
                    answer = number*10**(exponent)
                
                if answer < 1:
                   Result = Result.append({'Index': i, 'Predicted_Label': str(catR), 'Given_Label': y[i], 'ProbSpam':ProbSpam, 'ProbHam':ProbHam, 'expSpam':expSpam,'expHam':expHam}, ignore_index=True) 
                elif answer > 1:
                   Result = Result.append({'Index': i, 'Predicted_Label': str(catNR), 'Given_Label': y[i], 'ProbSpam':ProbSpam, 'ProbHam':ProbHam, 'expSpam':expSpam,'expHam':expHam}, ignore_index=True) 
                else:     
                    Result = Result.append({'Index': i, 'Predicted_Label': str(catR), 'Given_Label': y[i], 'ProbSpam':ProbSpam, 'ProbHam':ProbHam, 'expSpam':expSpam,'expHam':expHam}, ignore_index=True) 
      
        print(round(Files/float(NrMails)*100,4), '%')
        Result.to_csv('Result'+str(Dataset)+'-'+str(DocumentName),sep='\t')
    
    Timings = pd.DataFrame(columns = ['Description','time'])
    Timings = Timings.append(pd.Series({'Description':'Script', 'time': time.time()-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Classification', 'time': time.time()-start_time2}), ignore_index=True)
    Timings.to_csv('Timings'+str(Dataset)+'-'+str(DocumentName)+'Test',sep='\t')

l = ['TL1', 'TL2', 'TL3', 'TL4', 'TL5']
for batch in l:
    main(batch, 'MLGM'+str(batch), 0.5)