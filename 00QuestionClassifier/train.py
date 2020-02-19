from data_preprocess import Preprocess
import numpy as np
import torch
from bow_ffnn import BOW_FFNN

class Train:
    def __init__(self,tr_f,model,cfg_f,model_f):
        self.tr_f = tr_f
        self.model = model
        self.cfg_f = cfg_f
        self.model_f = model_f

    def train(self):
        read_config = open(self.cfg_f,'r')
        config_dict = dict()
        for line in read_config.readlines():
            split = line.split()
            config_dict[split[0]] = split[1]

        
        prpr = Preprocess(self.tr_f,self.cfg_f)
        # prpr.preprocess()
        labels,sentences,vocabulary,voca_embs,sens_rep,labels_index,labels_rep = prpr.load_preprocessed()
        # for sen_rep in sens_rep:
        #     print(sen_rep)
        # for element in prpr.load_preprocessed():
        #     for i in range(5):
        #         print(element[i])

        train_size = int(0.9*len(sens_rep))
        test_size = len(sens_rep)-train_size
        x_train,x_test = torch.utils.data.random_split(sens_rep,[train_size,test_size])
        y_train,y_test = torch.utils.data.random_split(labels_rep,[train_size,test_size])

        model = BOW_FFNN(voca_embs,int(config_dict['hidden_size']),len(labels_index),bool(config_dict['freeze']))