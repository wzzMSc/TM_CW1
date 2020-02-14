#%%
from stop_words import get_stop_words
stop_words = get_stop_words('english')
import pandas as pd
import re
#%%
input = open('train_5500.label','r')
punctuation = '!,;:?"\'`'

#%%
labels = list()
sentences = list()
for line in input.readlines():
    labels.append( line.split()[0] )
    # sentences.append( ' '.join(line.lower().split()[1:-1])+'\n' )
    sentence = line.lower().split()[1:-1]
    for word in sentence:
        if word in stop_words:
            sentence.remove(word)
    sentences.append(re.sub(r'[{}]+'.format(punctuation),'',' '.join(sentence)))

#%%
df_labels = pd.DataFrame(labels)
df_sentences = pd.DataFrame(sentences)

pd.concat([df_sentences,df_labels],axis=1).to_csv('merged.csv',header=False,index=False)