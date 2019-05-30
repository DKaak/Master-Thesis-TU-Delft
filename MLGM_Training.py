# -*- coding: utf-8 -*-
"""
python "c:\\users\\dkaak\\OneDrive - KPMG\\Documents\\Thesis\\Code\\Spam\\TrainingMLGM.py"
"""

import os
import pandas as pd
import time
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
    
    #%% 
    # =============================================================================
    # Start of training Generative Model 
    # =============================================================================
    Wordlist = list()
    SumLabel = pd.DataFrame(0, columns = [str(catR),str(catNR)], index = ['TotalSum'], dtype = 'uint32')
    Counter = pd.DataFrame(columns = [str(catR),str(catNR)], dtype = 'uint32')

    
    Batches = list(Traininglist.columns.values)
    Batches.remove(TestBatch)
    
    for batch in Batches:
        print(batch)
        for Files in range(int(SavePer),int(NrMails+SavePer),int(SavePer)):
            TrainingFile = pd.read_csv('Frequencies'+str(Dataset)+batch+'-'+str(Files),sep = '\t', index_col = 0)
            TrainingFile = TrainingFile.fillna(0)     
            
            Wordlist = list(set(Wordlist).union(set(TrainingFile.columns.values)))
            Wordlist.remove('Index_given')
            
            TrainingWords = list(TrainingFile.columns.values)
            TrainingWords.remove('Index_given')
            
            for j in [str(catR),str(catNR)]:
                SumLabel.loc['TotalSum',j] = SumLabel.loc['TotalSum',j] + float(TrainingFile.loc[TrainingFile['Index_given']==j][TrainingWords].sum().sum())
                PartialTraining = TrainingFile.loc[TrainingFile['Index_given']==j]
                for i in TrainingWords:
                    if i in list(Counter.index.values):
                        Counter.at[i,j] = Counter.at[i,j] + PartialTraining[i].sum()
                    else:
                        Counter = Counter.append(pd.Series({str(catR):0,str(catNR):0},name = i))
                        Counter.at[i,j] = Counter.at[i,j] + PartialTraining[i].sum()
            print(round(Files/float(NrMails)*100,4), '%')
    
    SumLabel = SumLabel + len(Wordlist)
    Counter = Counter+1
    
    Prob = pd.DataFrame(columns = ['Fraud','Legit'])
    for j in [str(catR), str(catNR)]:
        Prob[j] = Counter[j]/SumLabel[j][0]
    
    Counter.to_csv('Counter'+str(Dataset)+'-'+str(DocumentName)+str(TestBatch), sep = '\t')
    Prob.to_csv('Prob'+str(Dataset)+'-'+str(DocumentName)+str(TestBatch),sep='\t')
    
    Timings = pd.DataFrame(columns = ['Description','time'])
    Timings = Timings.append(pd.Series({'Description':'Script', 'time': time.time()-start_time}), ignore_index=True)
    Timings.to_csv('Timings'+str(Dataset)+'-'+str(DocumentName)+str(TestBatch)+'Training',sep='\t')



l = ['TL1', 'TL2', 'TL3', 'TL4', 'TL5']
for batch in l:
    main(str(batch), 'MLGM')

    
    
