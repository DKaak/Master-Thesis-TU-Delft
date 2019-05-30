# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:46:41 2019

@author: dkaak
"""

import os
from aspose.email.mapi import MapiMessage
import pandas as pd
import random
import time


def main():
    start_time = time.time()
    
    NrMails = int(17404)
    SavePer = int(40)
    
    random.seed(8370, version=1)
    rnlist = list()

    Traininglist = pd.DataFrame(columns = ['TL1','TL2', 'TL3', 'TL4', 'TL5'], index = range(1,100))
    
    os.chdir('C:\\Documents\\Test6 Random incl text files')
    
    Indexlist = pd.read_csv('Index.csv', sep=";", header=None, index_col=0)
    
    for i in range(1,NrMails+1):
        if i < 17302:
            msg = MapiMessage.from_file("ENRON ("+str(i)+").msg")
            sep2 = "EDRM Enron Email Data Set has been produced in"
            msg = msg.body.split(sep2,1)[0]
            msg = msg.replace('\n',' ').replace('\r',' ')
            
            msg = msg.lower().split(' ')    
            
            for m in range(0,len(msg)):
                for n in set(msg[m]):
                    if ord(n)>122: 
                        msg[m] = msg[m].replace(n,'')
                    elif ord(n)<97:
                        msg[m] = msg[m].replace(n,'')            
            uniqWords = sorted(set(msg)) 
            uniqWords.remove('')
        else:
            uniqWords = 'test'
        if len(uniqWords)>0:
            rnlist.append(i)  
                    
        
    for j in ['TL1', 'TL2', 'TL3', 'TL4', 'TL5']:
        for i in range(1,1625):
            rnumber = random.choice(rnlist) 
            Traininglist.at[i,j] = rnumber
            rnlist.remove(rnumber)
        
    Traininglist.to_csv('TraininglistENRON', sep = '\t')
    
    del i, j, rnumber, rnlist
    
    start_time2 = time.time()
    Freq = pd.DataFrame(columns = ['Index_given'])
    for l in list(Traininglist.columns.values):
        k=1
        for i in list(Traininglist[l]):
            if i < 17302:
                msg = MapiMessage.from_file("ENRON ("+str(i)+").msg")
                
                sep2 = "EDRM Enron Email Data Set has been produced in"
                msg = msg.body.split(sep2,1)[0]
                msg = msg.replace('\n',' ').replace('\r',' ')
                
                msg = msg.lower().split(' ')    
                
                for m in range(0,len(msg)):
                    for n in set(msg[m]):
                        if ord(n)>122: 
                            msg[m] = msg[m].replace(n,'')
                        elif ord(n)<97:
                            msg[m] = msg[m].replace(n,'')            
                uniqWords = sorted(set(msg))
                uniqWords.remove('')
                df = pd.DataFrame(columns = uniqWords)
                for word in uniqWords:
                    df.at[0,word] = msg.count(word)    
            else:
                msg = list()
                with open("ENRON ("+str(i)+").txt", 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.lower().replace('\n',' ').replace('\r', ' ').split()
                        for m in range(0,len(line)):
                            for n in set(line[m]):
                                if ord(n)>122: 
                                    line[m] = line[m].replace(n,'')
                                elif ord(n)<97:
                                    line[m] = line[m].replace(n,'')    
                        for word in line:
                            msg.append(word)
                uniqWords = sorted(set(msg))
                df = pd.DataFrame(columns = uniqWords)
                for word in uniqWords:
                    df.at[0,word] = msg.count(word)
                  
            Freq = Freq.append(df, ignore_index=True)
            if '' in Freq.columns:
                del Freq['']   
                    
            j = (k-1)%SavePer
            Freq.loc[j,'Index_given'] = Indexlist.loc[i][1]
            if k%SavePer == 0:
                Freq.to_csv('FrequenciesENRON'+l+'-'+str(k), sep = '\t')
                Freq = pd.DataFrame(columns = ['Index_given'])
                Freq = Freq.take(list())
            k=k+1 
        Freq.to_csv('FrequenciesENRON'+l+'-1640', sep = '\t')
        Freq = pd.DataFrame(columns = ['Index_given'])
        Freq = Freq.take(list())
            
    Timings = pd.DataFrame(columns = ['Description','time'])
    Timings = Timings.append(pd.Series({'Description':'Script', 'time': time.time()-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Traininglist', 'time': time.time()-start_time2}), ignore_index=True)
    Timings.to_csv('TimingsENRON-Setup',sep='\t')

if __name__ == '__main__':
    main()