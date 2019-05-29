# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:15:38 2019

@author: dkaak
"""

import os
import time
import pandas as pd

l = ['TL1', 'TL2', 'TL3', 'TL4', 'TL5']
for batch in l:
    main(batch)
    
def main(Testbatch):
    DocumentName = 'ENRON'+str(Testbatch)
    start_time = time.time()
    os.chdir('C:\\Documents\\Test6 Random incl text files')
    Indexlist = pd.read_csv('Index.csv', sep=";", header=None)
    Indexlist.columns = ['Index', 'document']
    
    #%%
    NrMails = 1640 #int(raw_input('Enter number of mails (only per 1000): '))
    SavePer = 40 #int(raw_input('Save per number of mails (only per 100): '))
    
    Traininglist = pd.read_csv('TraininglistENRON',sep = '\t', index_col = 0)
    
    
    Wordlist = list()
    SumLabel = pd.DataFrame(0, columns = ['Fraud','Legit'], index = ['TotalSum'], dtype = 'uint32')
    Counter = pd.DataFrame(columns = ['Fraud','Legit'], dtype = 'uint32')  
    Batches = list(Traininglist.columns.values)
    Batches.remove(Testbatch)
    for batch in Batches:
        print(batch)
        for Files in range(int(SavePer),int(NrMails+SavePer),int(SavePer)):
            TrainingFile = pd.read_csv('FrequenciesENRON'+batch+'-'+str(Files),sep = '\t', index_col = 0)
            TrainingFile = TrainingFile.fillna(0)        
            Indexlist = TrainingFile['Index_given']
            
            Wordlist = list(set(Wordlist).union(set(TrainingFile.columns.values)))
            Wordlist.remove('Index_given')
            
            TrainingWords = list(TrainingFile.columns.values)
            TrainingWords.remove('Index_given')
            
            for j in ['Fraud','Legit']:
                SumLabel.loc['TotalSum',j] = SumLabel.loc['TotalSum',j] + float(TrainingFile.loc[TrainingFile['Index_given']==j][TrainingWords].sum().sum())
                PartialTraining = TrainingFile.loc[TrainingFile['Index_given']==j]
                for i in TrainingWords:
                    if i in list(Counter.index.values):
                        Counter.at[i,j] = Counter.at[i,j] + PartialTraining[i].sum()
                    else:
                        Counter = Counter.append(pd.Series({'Fraud':0,'Legit':0},name = i))
                        Counter.at[i,j] = Counter.at[i,j] + PartialTraining[i].sum()
            print(round(Files/float(NrMails)*100,4), '%')
    SumLabel = SumLabel + len(Wordlist)
    Counter = Counter+1
    Prob = pd.DataFrame(columns = ['Fraud','Legit'])
    for j in ['Fraud', 'Legit']:
        Prob[j] = Counter[j]/SumLabel[j][0]
    Prob['Ratio'] = Prob['Fraud']/Prob['Legit']
    Prob = Prob.sort_values(by='Ratio')
    spamwords = list(Prob.index[-500:])
    hamwords = list(Prob.index[0:500])
    wordselection = spamwords+hamwords
    wordselection = pd.DataFrame({'words':wordselection})
    wordselection.to_csv('wordselection-'+str(DocumentName),sep='\t')