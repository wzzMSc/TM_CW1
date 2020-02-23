from stop_words import get_stop_words
import pandas as pd
import re
from gensim.models import Word2Vec as w2v
from gensim.models import KeyedVectors as kv
import numpy as np


class Preprocess:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        stop_words = get_stop_words('english')


        stop_words.remove("what")
        stop_words.remove("how")
        stop_words.remove("who")
        stop_words.remove("where")
        stop_words.remove("which")
        stop_words.remove("why")
        input = open(self.data, 'r')
        punctuation = '!,;:?"\'`'


        labels = list()
        sentences = list()

        for line in input.readlines():
            labels.append(line.split()[0])
            sentence = line.lower().split()[1:]
            for word in line.lower().split()[1:]:
                if word in stop_words:
                    sentence.remove(word)
            sentences.append(re.sub(r'[{}]+'.format(punctuation), '', ' '.join(sentence)))

        return labels,sentences

    def get_label_index(self,labels):
        labels_count_dict = {}
        for label in labels:
            if label in labels_count_dict:
                labels_count_dict[label] += 1
            else:
                labels_count_dict[label] = 1
        labels_index_dict = {}
        index = 0
        for label in list(labels_count_dict.items()).sort(key=lambda x:x[1],reverse=True):
            labels_index_dict[label[0]] = index
            index += 1
        return labels_index_dict

    def get_word_count(self,sentences):
        words_count_dict = {}
        
        for sentence in sentences:
            for word in sentence.split():
                if word in words_count_dict:
                    words_count_dict[word] += 1
                else:
                    words_count_dict[word] = 1
        return list(words_count_dict.items()).sort(key=lambda x:x[1],reverse=True)

    def get_bow(self,sentences,pretrained_path):
        emb = kv.load_word2vec_format(pretrained_path,binary=True)
        vecs_list = list()
        for sentence in sentences:
            word_list = sentence.split()
            vec = np.zeros(300)
            for word in word_list:
                vec += emb[word]
            vecs_list.append(vec/len(word_list))
        return vecs_list

    def get_indexed_labels(self,labels,labels_index_dict):
        indexed_labels = list()
        for label in labels:
            indexed_labels.append(labels_index_dict[label])
        return indexed_labels