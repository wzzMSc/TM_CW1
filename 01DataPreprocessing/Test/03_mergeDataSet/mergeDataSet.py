# import os
# os.system("pip install pandas")
import pandas as pd
import re

punctuation = '!,;:?"\'`'
sentences = open('train_5500_sentences_noSW.txt','r')
sentences_noPunc = open('train_5500_sentences_noSW_noPunc.txt','w')
outlines = list()
for line in sentences.readlines():
    outlines.append(re.sub(r'[{}]+'.format(punctuation),'',line))

sentences_noPunc.writelines(outlines)
sentences.close()
sentences_noPunc.close()

df_label = pd.read_csv('train_5500_labels.txt')
df_sentences = pd.read_csv('train_5500_sentences_noSW_noPunc.txt')

pd.concat([df_sentences,df_label],axis=1).to_csv('merged.csv',header=False,index=False)