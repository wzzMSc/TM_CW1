from stop_words import get_stop_words
import pandas as pd
import re
from gensim.models import Word2Vec as w2v
from gensim.models import KeyedVectors as kv
import numpy as np


class Preprocess:
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def preprocess(self):
        stop_words = get_stop_words('english')


        stop_words.remove("what")
        stop_words.remove("how")
        stop_words.remove("who")
        stop_words.remove("where")
        stop_words.remove("which")
        stop_words.remove("why")
        punctuation = '!,;:?"\'`'

        
        input = open(self.data, 'r')


        labels = list()
        sentences = list()

        for line in input.readlines():
            labels.append(line.split()[0])
            sentence = line.lower().split()[1:]
            for word in line.lower().split()[1:]:
                if word in stop_words:
                    sentence.remove(word) # remove stop words
            sentences.append(re.sub(r'[{}]+'.format(punctuation), '', ' '.join(sentence))) # remove punctuation

        words_count_dict = {}
        
        for sentence in sentences:
            for word in sentence.split():
                if word in words_count_dict:
                    words_count_dict[word] += 1
                else:
                    words_count_dict[word] = 1
        vocabulary = list()
        for word_count in list(words_count_dict.items()).sort(key=lambda x:x[1],reverse=True):
            if word_count[1]>self.config['min_words']:
                vocabulary.append(word_count[0]) # form vocabulary in the order of word count
        
        emb = kv.load_word2vec_format(self.config['word_embeddings_path'],binary=True)
        voca_emb = list()
        for word in vocabulary:
            voca_emb.append(emb[word]) # extract word embeddings in the order of vocabulary
        voca_emb.append(np.mean(voca_emb,axis=0)) # embedding for unknown word

        sens_rep = list()
        for sentence in sentences:
            sen_rep = np.zeros(len(vocabulary))
            for word in sentence.split():
                if word in vocabulary:
                    sen_rep[vocabulary.index(word)] = 1 # in form of (0,1,0,0,1,...)

        

