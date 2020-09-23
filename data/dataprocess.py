#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:shiran
# datetime:2020/9/22 3:12 PM
# software: PyCharm


import codecs
import re
import pdb
import pandas as pd
import  numpy as np
import collections
from collections.abc import Iterable

def decreas_data():
    i=0
    with open('./renmin.txt','r')as inp,open('./renmino.txt','w')as outp:
        i=0
        for line in inp.readlines():
            if(i<1000):
                i+=1
                outp.write(line)
            else:
                break
            # if i<10000:
            #     outp.write(inp.read())
            #     i+=1
            # else:
            #     pass




def process_data():       #将数据中的首行数据进行删除
    with open('./renmino.txt','r')as inp,open('./renmin2.txt','w')as outp:
        for line in inp.readlines():
            line=line.split('  ')
            i=1
            while i<len(line)-1:
                if line[i][0]=='[':
                    outp.write(line[i].split('/')[0][1:])
                    i+=1
                    while i<len(line)-1 and line[i].find(']')==-1:
                        if line[i]!='':
                            outp.write(line[i].split('/')[0])
                        i+=1
                    outp.write(line[i].split('/')[0].strip()+'/'+line[i].split('/')[1][-2:]+' ')
                elif line[i].split('/')[1]=='nr':
                    word=line[i].split('/')[0]
                    i+=1
                    if i<len(line)-1 and line[i].split('/')[1]=='nr':
                        outp.write(word+line[i].split('/')[0]+'/nr ')
                    else:
                        outp.write(word+'/nr ')
                        continue
                else:
                    outp.write(line[i]+' ')
                i+=1
            outp.write('\n')


def process_data2():            #将数据中的标签转化为B_,M_,E_的标签序列
    with codecs.open('./renmin2.txt','r','utf-8')as inp,codecs.open('./renmin3.txt','w','utf-8')as outp:
        for line in inp.readlines():
            line=line.split(' ')
            i=0
            while i<len(line)-1:
                if line[i]=='':
                    i+=1
                    continue
                word=line[i].split('/')[0]
                tag=line[i].split('/')[1]
                if tag=='nr' or tag=='ns' or tag=='nt':
                    outp.write(word[0]+"/B_"+tag+" ")
                    for j in word[1:len(word)-1]:
                        if j!='':
                            outp.write(j+"/M_"+tag+" ")
                    outp.write(word[-1]+"/E_"+tag+" ")
                else:
                    for wor in word:
                        outp.write(wor+'/o ')
                i+=1
            outp.write('\n')
def process_data3_sentence():             #将renmin3中的数据进行
    with open('./renmin3.txt','r')as inp,codecs.open('./renmin4.txt','w','utf-8')as outp:
        texts=inp.read()
        sentences=re.split('[，。!?、''"":]/[O]',texts)
        for sentence in sentences:
            if sentence!=" ":
                outp.write(sentence.strip()+'\n')

import collections
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def data2pkl():
    datas=list()
    labels=list()
    linedata=list()
    linelabel=list()
    tags=set()
    tags.add('')
    input_data=codecs.open('renmin4.txt','r','utf-8')
    for line in input_data.readlines():
        line=line.split()
        linedata=[]
        linelabel=[]
        numNoto=0
        for word in line:
            word=word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1]!='o':
                numNoto+=1
            if numNoto!=0:
                datas.append(linedata)
                labels.append(linelabel)
    input_data.close()
    print("the total words data number is:",len(datas))
    print("the total  label number is:",len(labels))
    all_words=flatten(datas)
    sr_allwords=pd.Series(all_words)
    sr_allwords=sr_allwords.value_counts()
    set_words=sr_allwords.index
    set_ids=range(1,len(set_words)+1)
    print("it work well")
    tags=[i for i in tags]
    tag_ids=range(len(tags))
    word2id=pd.Series(set_ids,index=set_words)
    id2word=pd.Series(set_words,index=set_ids)
    tag2id=pd.Series(tag_ids,index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id["unknow"]=len(word2id)+1
    id2word[len(word2id)]="unknow"
    print(tag2id)
    max_len=60
    def X_padding(words):
        ids=list(word2id[words])
        if len(ids)>=max_len:
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids)))
        return ids

    def y_padding(tags):
        ids=list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids)))
        return ids
    df_data=pd.DataFrame({'words':datas,'tags':labels},index=range(len(datas)))
    df_data['x']=df_data['words'].apply(X_padding)
    df_data['y']=df_data['tags'].apply(y_padding)
    x=np.asanyarray(list(df_data['x'].values))
    y=np.asanyarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)
    x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.2,random_state=43)


    import pickle
    import os
    with open('../renmindata.pkl','wb')as outp:
        pickle.dump(word2id,outp)
        pickle.dump(id2word,outp)
        pickle.dump(tag2id,outp)
        pickle.dump(id2tag,outp)
        pickle.dump(x_train,outp)
        pickle.dump(y_train,outp)
        pickle.dump(x_test,outp)
        pickle.dump(y_test,outp)
        pickle.dump(x_valid,outp)
        pickle.dump(y_valid,outp)


    print('*****Finished saving the data')

decreas_data()
process_data()
process_data2()
process_data3_sentence()
data2pkl()