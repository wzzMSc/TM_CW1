# import os
# os.system("pip install stop-words")
# os.system("pip install pandas")
from stop_words import get_stop_words
import pandas as pd
import re


stop_words = get_stop_words('english')
stop_words.remove("what")
stop_words.remove("how")
stop_words.remove("who")
stop_words.remove("where")
stop_words.remove("which")
input = open('train_5500.label','r')
punctuation = '!,;:?"\'`'


labels = list()
sentences = list()
labels.append('Labels')
sentences.append('Sentences')

for line in input.readlines():
    labels.append( line.split()[0] )
    sentence = line.lower().split()[1:]
    sentence_iterator = line.lower().split()[1:]
    for word in sentence_iterator:
        if word in stop_words:
            sentence.remove(word)
    sentences.append(re.sub(r'[{}]+'.format(punctuation),'',' '.join(sentence)))


df_labels = pd.DataFrame(labels)
df_sentences = pd.DataFrame(sentences)

pd.concat([df_sentences,df_labels],axis=1).to_csv('merged.csv',header=False,index=False)