# import os
# os.system("pip install gensim")
from gensim.models import Word2Vec as w2v
from gensim.models import KeyedVectors as kv

# import os
# os.system("pip install pandas")
import pandas as pd

# import os
# os.system("pip install numpy")
import numpy as np

df = pd.read_csv('merged.csv')
sens_list = df.iloc[:,0].tolist()

vectors = kv.load_word2vec_format('pretrained.vector')

# print(type(vectors['the']))
vecs_list = list()
vecs_list.append('Vectors')


for sentence in sens_list:
    word_list = sentence.split()
    vec = np.zeros(100)
    for word in word_list:
        vec += vectors[word]
    vecs_list.append(vec/len(word_list))


sens_list.insert(0,'Sentences')

df_sens = pd.DataFrame(sens_list)
df_vecs = pd.DataFrame(vecs_list)
pd.concat([df_sens,df_vecs],axis=1).to_csv('sens_vecs.csv',header=False,index=False)