# -*- coding: utf-8 -*-

import os
import pandas as pd
import time
import numpy as np
import copy
import sys

#%%
def main(TestBatch, DocumentName, p):
    p = float(p)
    start_time = time.time()
    
    Dataset = 'ENRON' #'TREC'
    catR = 'Fraud' #'spam'
    catNR = 'Legit' #'ham'
    
    NrMails = 1640 #75000/5
    SavePer = 40 #100
    
    #%%
    Counter = pd.read_csv('Counter'+str(Dataset)+'-'+str(DocumentName), sep='\t', index_col = 0)
    Traininglist = pd.read_csv('Traininglist'+str(Dataset),sep = '\t', index_col = 0)
    
    #%%
    # =============================================================================
    # Applying Bayesian Model to test data
    # =============================================================================
    start_time2 = time.time()
    
    factdictBase = {}
    factdictExp = {}
    answer = 1.0
    factdictBase[0] = 1.0
    factdictExp[0]= 0.0
    factdictBase[1] = 1.0
    factdictExp[1]= 0.0
    for i in range(2,10000000):
        answer = factdictBase[i-1]*i
        factdictExp[i] = factdictExp[i-1] + np.floor(np.log10(np.abs(answer)))
        factdictBase[i] = answer/(10**abs(np.floor(np.log10(np.abs(answer))))) 
    
    alpha = {}
    alphaySpam = {}
    GalphayBaseSpam = {}
    GalphayExpSpam = {}
    alphayHam = {}
    GalphayBaseHam = {}
    GalphayExpHam = {}
    
    totalwordlist1 = list(Counter.index)
    totalwordlist = list()
    for word in totalwordlist1:
        if type(word)==str:
            totalwordlist.append(word)
    
    for word in totalwordlist:
        alpha[word] = 1
        alphaySpam[word] = 0
        GalphayBaseSpam[word] = 0.0
        GalphayExpSpam[word] = 0.0
        alphayHam[word] = 0
        GalphayBaseHam[word] = 0.0
        GalphayExpHam[word] = 0.0
    
    fac2spam = 1
    fac2spamExp = 0
    fac2ham = 1
    fac2hamExp = 0
    
    for word in totalwordlist:
        alphaySpam[word] = Counter[str(catR)][word] + alpha[word] 
        alphayHam[word] = Counter[str(catNR)][word] + alpha[word]
        GalphayBaseSpam[word], GalphayExpSpam[word] = factdictBase[alphaySpam[word]-1], factdictExp[alphaySpam[word]-1]
        GalphayBaseHam[word], GalphayExpHam[word] = factdictBase[alphayHam[word]-1], factdictExp[alphayHam[word]-1]
    
        fac2spam = fac2spam*GalphayBaseSpam[word]
        fac2spamExp = fac2spamExp + GalphayExpSpam[word] + np.floor(np.log10(np.abs(fac2spam)))
        fac2spam = fac2spam/(10**abs(np.floor(np.log10(np.abs(fac2spam)))))
        
        fac2ham = fac2ham*GalphayBaseHam[word]
        fac2hamExp = fac2hamExp + GalphayExpHam[word] + np.floor(np.log10(np.abs(fac2ham)))
        fac2ham = fac2ham/(10**abs(np.floor(np.log10(np.abs(fac2ham)))))  
    
    Result = pd.DataFrame(columns = ['Index', 'Predicted_Label','Given_Label'])
    batch = TestBatch
    
    start_time3 = time.time()
    for Files in range(int(SavePer),int(NrMails+SavePer),int(SavePer)):
        Freq2 = pd.DataFrame() 
        Freq2 = pd.read_csv('Frequencies'+str(Dataset)+batch+'-'+str(Files),sep = '\t', index_col=0)
        Freq2 = Freq2.fillna(0)
        GivenLabel = Freq2['Index_given']
        del Freq2['Index_given']
        
        NrPredictions = len(Freq2)
        for i in range(0,NrPredictions):
            GalphaysimBaseSpam = copy.copy(GalphayBaseSpam)
            GalphaysimExpSpam = copy.copy(GalphayExpSpam)
            GalphaysimBaseHam = copy.copy(GalphayBaseHam)
            GalphaysimExpHam = copy.copy(GalphayExpHam)

            
            MessageWords = list(Freq2.columns[Freq2.iloc[i]>0])
            if len(MessageWords)>0:
                fac1spam = 1
                fac1spamExp = 0
                        
                fac1ham = 1 
                fac1hamExp = 0
                
                sum1spam = 0
                sum1ham = 0 
                
                fac4counter = 0
                
                CommonWords = list(set(MessageWords)&set(totalwordlist))
                NewWords = list(set(MessageWords)-set(totalwordlist))
                for word in CommonWords:
                    GalphaysimBaseSpam[word], GalphaysimExpSpam[word] = factdictBase[alphaySpam[word]+Freq2[word][i]-1], factdictExp[alphaySpam[word]+Freq2[word][i]-1]
                    GalphaysimBaseHam[word], GalphaysimExpHam[word] = factdictBase[alphayHam[word]+Freq2[word][i]-1], factdictExp[alphayHam[word]+Freq2[word][i]-1]
                
                for word in NewWords:
                    base, exp = factdictBase[1+Freq2[word][i]-1], factdictExp[1+Freq2[word][i]-1]
                    GalphaysimBaseSpam[word] = base
                    GalphaysimExpSpam[word] = exp
                    GalphaysimBaseHam[word] = base
                    GalphaysimExpHam[word] = exp
                    fac4counter = fac4counter + 1            
        
                fac4spam, fac4spamExp = factdictBase[sum(alphaySpam.values())+fac4counter-1], factdictExp[sum(alphaySpam.values())+fac4counter-1]
                fac4ham, fac4hamExp = factdictBase[sum(alphayHam.values())+fac4counter-1], factdictExp[sum(alphayHam.values())+fac4counter-1]
                
                start_time4 = time.time()
                DictionaryWords = list(set(GalphaysimBaseSpam.keys())-set(NewWords)-set(CommonWords))
                for word in DictionaryWords:
                    answerbase = GalphaysimBaseSpam[word]
                    answerexp = GalphaysimExpSpam[word]
                    fac1spam = fac1spam*answerbase
                    fac1spamExp = fac1spamExp + answerexp + np.floor(np.log10(np.abs(fac1spam)))
                    fac1spam = fac1spam/(10**abs(np.floor(np.log10(np.abs(fac1spam)))))
                    
                    answerbase = GalphaysimBaseHam[word]
                    answerexp = GalphaysimExpHam[word]
                    fac1ham = fac1ham*answerbase
                    fac1hamExp = fac1hamExp + answerexp + np.floor(np.log10(np.abs(fac1ham)))
                    fac1ham = fac1ham/(10**abs(np.floor(np.log10(np.abs(fac1ham)))))      
                    
                    sum1spam = sum1spam + alphaySpam[word] 
                    sum1ham = sum1ham + alphayHam[word]           
                
                for word in NewWords:
                    answerbase = GalphaysimBaseSpam[word]
                    answerexp = GalphaysimExpSpam[word]
                    fac1spam = fac1spam*answerbase
                    fac1spamExp = fac1spamExp + answerexp + np.floor(np.log10(np.abs(fac1spam)))
                    fac1spam = fac1spam/(10**abs(np.floor(np.log10(np.abs(fac1spam)))))
                    
                    answerbase = GalphaysimBaseHam[word]
                    answerexp = GalphaysimExpHam[word]
                    fac1ham = fac1ham*answerbase
                    fac1hamExp = fac1hamExp + answerexp + np.floor(np.log10(np.abs(fac1ham)))
                    fac1ham = fac1ham/(10**abs(np.floor(np.log10(np.abs(fac1ham)))))      
                    
                    sum1spam = sum1spam + Freq2[word][i] + 1
                    sum1ham = sum1ham + Freq2[word][i]+ 1       
                    
                for word in CommonWords:
                    answerbase = GalphaysimBaseSpam[word]
                    answerexp = GalphaysimExpSpam[word]
                    fac1spam = fac1spam*answerbase
                    fac1spamExp = fac1spamExp + answerexp + np.floor(np.log10(np.abs(fac1spam)))
                    fac1spam = fac1spam/(10**abs(np.floor(np.log10(np.abs(fac1spam)))))
                    
                    answerbase = GalphaysimBaseHam[word]
                    answerexp = GalphaysimExpHam[word]
                    fac1ham = fac1ham*answerbase
                    fac1hamExp = fac1hamExp + answerexp + np.floor(np.log10(np.abs(fac1ham)))
                    fac1ham = fac1ham/(10**abs(np.floor(np.log10(np.abs(fac1ham)))))      
                    
                    sum1spam = sum1spam + alphaySpam[word] + Freq2[word][i]
                    sum1ham = sum1ham + alphayHam[word] + Freq2[word][i]       
                
                
                fac3spam, fac3spamExp  = factdictBase[sum1spam-1], factdictExp[sum1spam-1]
                fac3ham, fac3hamExp  = factdictBase[sum1ham-1],factdictExp[sum1ham-1]
                
                ProbSpam = fac1spam/fac2spam*fac4spam/fac3spam*float(p)
                ProbHam = fac1ham/fac2ham*fac4ham/fac3ham*float(1-p)
                
                expSpam = fac1spamExp-fac2spamExp+fac4spamExp-fac3spamExp
                expHam = fac1hamExp-fac2hamExp+fac4hamExp-fac3hamExp
                
                del GalphaysimBaseSpam
                del GalphaysimExpSpam
                del GalphaysimBaseHam
                del GalphaysimExpHam
                
                while ProbSpam>1:
                    ProbSpam = ProbSpam/10
                    expSpam = expSpam-1
                while ProbHam>1:
                    ProbHam = ProbHam/10
                    expHam = expHam-1
                
                while ProbSpam<0.1:
                    ProbSpam = ProbSpam*10
                    expSpam = expSpam+1
                while ProbHam<0.1:
                    ProbHam = ProbHam*10
                    expHam = expHam+1  
                
                number = ProbHam/ProbSpam
                exponent = expHam-expSpam
                    
                if number <1 and exponent > 100:
                    answer = 10
                elif number > 1 and exponent > 100:
                    answer = 10
                else:
                    answer = number*10**(exponent)
                
                if answer < 1:
                   Result = Result.append({'Index': i, 'Predicted_Label': str(catR), 'Given_Label': GivenLabel[i], 'ProbSpam':ProbSpam, 'ProbHam':ProbHam, 'expSpam':expSpam,'expHam':expHam}, ignore_index=True) 
                elif answer > 1:
                   Result = Result.append({'Index': i, 'Predicted_Label': str(catNR), 'Given_Label': GivenLabel[i], 'ProbSpam':ProbSpam, 'ProbHam':ProbHam, 'expSpam':expSpam,'expHam':expHam}, ignore_index=True) 
                else:     
                    Result = Result.append({'Index': i, 'Predicted_Label': str(catR), 'Given_Label': GivenLabel[i], 'ProbSpam':ProbSpam, 'ProbHam':ProbHam, 'expSpam':expSpam,'expHam':expHam}, ignore_index=True) 
            
        print(round(Files/float(NrMails)*100,4), '%')
    
    Result.to_csv('Result'+str(Dataset)+'-'+str(DocumentName),sep='\t')
    
    Timings = pd.DataFrame(columns = ['Description','time'])
    Timings = Timings.append(pd.Series({'Description':'Script', 'time': time.time()-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Intro', 'time': start_time2-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Initialization', 'time': start_time3-start_time2}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Classification', 'time': time.time()- start_time3}), ignore_index=True)
    Timings.to_csv('Timings'+str(Dataset)+'-'+str(DocumentName)+'Test',sep='\t')

    
l = ['TL1', 'TL2', 'TL3', 'TL4', 'TL5']
for batch in l:
    main(batch, 'BEGM'+str(batch), 0.5)