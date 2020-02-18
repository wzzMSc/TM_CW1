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
        stop_words.remove("when")
        punctuation = '!,;:?"\'`'

        
        input = open(self.data, 'r')


        labels = list()
        sentences = list()

        for line in input.readlines():
            labels.append(line.split()[0])
            sentence = line.lower().split()[1:]
            # print(sentence)
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
        # print(words_count_dict)
        vocabulary = list()
        wcd_sorted = list(words_count_dict.items())
        wcd_sorted.sort(key=lambda x:x[1],reverse=True)

        read_config = open(self.config,'r')
        config_dict = dict()
        for line in read_config.readlines():
            split = line.split()
            config_dict[split[0]] = split[1]

        # print(type(config_dict))
        for word_count in wcd_sorted:
            if word_count[1]>int(config_dict['min_words']):
                vocabulary.append(word_count[0]) # form vocabulary in the order of word count
        
        emb = kv.load_word2vec_format(config_dict['word_embeddings_path'],binary=True)
        voca_embs = list()
        for word in vocabulary:
            try:
                voca_embs.append(emb[word]) # extract word embeddings in the order of vocabulary
            except KeyError:
                voca_embs.append("NaN")


        voca_emb_avg_list = list()
        for voca_emb in voca_embs:
            if voca_emb != "NaN":
                voca_emb_avg_list.append(voca_emb)
        voca_emb_avg = np.mean(voca_emb_avg_list,axis=0)
        for i in range(len(voca_embs)):
            if voca_embs[i] == "NaN":
                voca_embs[i] = voca_emb_avg
        
        voca_embs.append(voca_emb_avg)

        sens_rep = list()
        # for sentence in sentences:
        #     sen_rep = np.zeros(len(vocabulary))
        #     for word in sentence.split():
        #         if word in vocabulary:
        #             sen_rep[vocabulary.index(word)] = 1 # in form of (0,1,0,0,1,...)
        for sentence in sentences:
            sen_rep = list()
            for word in sentence.split():
                if word in vocabulary:
                    sen_rep.append(vocabulary.index(word))
                else:
                    sen_rep.append(len(voca_embs)-1)
            sens_rep.append(sen_rep)

        # for vo in vocabulary:
        #     print(vo)
        # for rep in sens_rep:
        #     print(rep)
        # print("stop")

        pd.DataFrame(labels).to_csv('labels.csv',header=False,index=False)
        pd.DataFrame(sentences).to_csv('sentences.csv',header=False,index=False)
        pd.DataFrame(vocabulary).to_csv('vocabulary.csv',header=False,index=False)
        pd.DataFrame(voca_embs).to_csv('voca_embs.csv',header=False,index=False)
        pd.DataFrame(sens_rep).to_csv('sens_rep.csv',header=False,index=False)

        

