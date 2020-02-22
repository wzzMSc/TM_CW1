from stop_words import get_stop_words
import pandas as pd
import re
# from gensim.models import Word2Vec as w2v
# from gensim.models import KeyedVectors as kv
import numpy as np
import pickle


class Preprocess:
    def __init__(self, config):
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
        punctuation = '!,.;:?"\'`'

        
        input = open(self.config["path_data"], 'r')


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
        wcd_sorted = list(words_count_dict.items())
        wcd_sorted.sort(key=lambda x:x[1],reverse=True)

        for word_count in wcd_sorted:
            if word_count[1]>int(self.config['min_words']):
                vocabulary.append(word_count[0]) # form vocabulary in the order of word count
        
        # emb = kv.load_word2vec_format(self.config['word_embeddings_path'],binary=True)
        # voca_embs = list()
        # for word in vocabulary:
        #     try:
        #         voca_embs.append(emb[word].tolist()) # extract word embeddings in the order of vocabulary
        #     except KeyError:
        #         voca_embs.append("NaN")

        emb = dict()
        with open(self.config['path_pre_emb'],'r') as f:
            for line in f.readlines():
                line_list = line.split()
                emb[line_list[0]] = line_list[1:]
        voca_embs = list()
        for word in vocabulary:
            try:
                voca_embs.append(emb[word])
            except KeyError:
                voca_embs.append(emb['#UNK#'])        
        voca_embs.append(emb['#UNK#'])

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

        labels_index = dict()
        count=0
        for label in labels:
            if label not in labels_index:
                labels_index[label] = count
                count += 1
        labels_rep = list()
        for label in labels:
            labels_rep.append(labels_index[label])



        self.save(labels,'../data/labels.bin')
        self.save(sentences,'../data/sentences.bin')
        self.save(vocabulary,'../data/vocabulary.bin')
        self.save(voca_embs,'../data/voca_embs.bin')
        self.save(sens_rep,'../data/sens_rep.bin')
        self.save(list(labels_index.items()),'../data/labels_index.bin')
        self.save(labels_rep,'../data/labels_rep.bin')
        

        
    def load_preprocessed(self):
        return self.load('../data/labels.bin'),self.load('../data/sentences.bin'),self.load('../data/vocabulary.bin'),self.load('../data/voca_embs.bin'),self.load('../data/sens_rep.bin'),self.load('../data/labels_index.bin'),self.load('../data/labels_rep.bin')

    def save(self,l,f_name):
        pickle.dump(l,open(f_name,'wb'))

    def load(self,f_name):
        return pickle.load(open(f_name,'rb'))