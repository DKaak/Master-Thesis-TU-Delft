# -*- coding: utf-8 -*-
"""
Cleaning and count words in Spam Dataset Trec07p
"""
import os
import email
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def main():
    start_time = time.time()

    os.chdir('C:\\Documents\\Spam\\trec07p\\full')
    Indexlist = pd.read_csv('index', sep=" ", header=None)
    Indexlist.columns = ['Index', 'document']
    
    os.chdir('C:\\Documents\\Spam\\trec07p\\data')
    
    #%%
    NrMails = int(75000)
    SavePer = int(100)
    
    
    #%%
    random.seed(1998, version=1)
    rnlist = list()
    
    Traininglist = pd.DataFrame(columns = ['TL1','TL2','TL3','TL4','TL5'], index = range(1,int(NrMails/5)+1))
    
    for i in range(1,NrMails+1):
        rnlist.append(i)
    
    for j in list(Traininglist.columns.values):
        for i in range(1,int(NrMails/5+1)):
            rnumber = random.choice(rnlist) 
            Traininglist.at[i,j] = rnumber
            rnlist.remove(rnumber)
    
    Traininglist.to_csv('TraininglistTREC', sep = '\t')
    del i, j, rnumber
    
    #%% 
    # =============================================================================
    # Start of ANALYSIS TREC data
    # =============================================================================
    start_time2 = time.time()
    def findmessage(msg, subtype):
        if subtype == 'html':
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    soup = BeautifulSoup(part.get_payload(decode=False), 'html.parser')
                    tekst = soup.get_text().replace('\n',' ').replace('\t',' ').replace(u'\xa0',' ')
                    message = tekst.lower().split(' ') 
                    break
        elif subtype == 'plain':
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    tekst = part.get_payload().replace('\n',' ').replace('\t',' ')
                    message = tekst.lower().split(' ')   
                    break
        else: 
            message = ' '
    
        return message
    
    Freq = pd.DataFrame(columns = ['Index_given'])
    for l in list(Traininglist.columns.values):
        k = 1
        for i in list(Traininglist[l]):
            testfile = open('inmail.'+str(i),'r', encoding="latin1")  
            msg = email.message_from_file(testfile)
            
            typemainlist = list()
            typesublist = list()
            for part in msg.walk():
                typemainlist.append(part.get_content_maintype())
                typesublist.append(part.get_content_subtype())
        
            message = ''
            if 'text' in typemainlist:
                if 'html' in typesublist:
                    message = findmessage(msg,'html')
                elif 'plain' in typesublist: 
                    message = findmessage(msg,'plain')
                else: 
                    print('Not implemented (4)', i)
            else:
                print('Not implemented (5)', i)
            for m in range(0,len(message)):
                for n in message[m]:
                    if ord(n)>122:
                        message[m] = message[m].replace(n,'')
                    elif ord(n)<97:
                        message[m] = message[m].replace(n,'')
                        
            uniqWords = sorted(set(message))
            df = pd.DataFrame(columns = uniqWords)
            for word in uniqWords:
                df.at[0,word] = message.count(word)
              
            Freq = Freq.append(df, ignore_index=True)
            if '' in Freq.columns:
                del Freq['']
            j = (k-1)%SavePer
            Freq.loc[j,'Index_given'] = Indexlist.loc[i-1,'Index']
            if k%SavePer == 0:
                Freq.to_csv('FrequenciesTREC'+l+'-'+str(k), sep = '\t')
                Freq = pd.DataFrame(columns = ['Index_given'])
                Freq = Freq.take(list())
            k=k+1
            
    start_time4 = time.time()        
    
    Timings = pd.DataFrame(columns = ['Description','time'])
    Timings = Timings.append(pd.Series({'Description':'Script', 'time': time.time()-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Initialization', 'time': start_time-start_time}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Traininglist', 'time': start_time4-start_time2}), ignore_index=True)
    Timings = Timings.append(pd.Series({'Description':'Testlist', 'time': time.time()-start_time4}), ignore_index=True)
    Timings.to_csv('Timings-Setup',sep='\t')
    
if __name__ == '__main__':
    main()