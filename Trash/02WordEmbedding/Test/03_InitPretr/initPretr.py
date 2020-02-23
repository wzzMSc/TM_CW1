# import os
# os.system("pip install gensim")
from gensim.models import Word2Vec

# import os
# os.system("pip install pandas")
import pandas as pd


df = pd.read_csv('merged.csv')
sentences = df.iloc[:,0].tolist()
input = list()
for sentence in sentences:
    input.append(sentence.split())
model = Word2Vec(input, min_count=1,workers=12)
model.wv.save_word2vec_format('pretrained.vector')