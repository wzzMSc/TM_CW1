from data_preprocess import Preprocess
import numpy as np
import torch
from bow_ffnn import BOW_FFNN
from qc_dataset import QCDataset
from torch.utils.data.dataloader import DataLoader

class Train:
    def __init__(self,config):
        self.config = config

    def train(self):

        
        prpr = Preprocess(self.config)
        # prpr.preprocess()
        labels,sentences,vocabulary,voca_embs,sens_rep,labels_index,labels_rep = prpr.load_preprocessed()
        # for element in prpr.load_preprocessed():
        #     for i in range(5):
        #         print(element[i])

        train_size = int(0.8*len(sens_rep))
        dev_size = int(0.1*len(sens_rep))
        test_size = len(sens_rep)-train_size-dev_size

        merged_rep = list()
        for i in range(len(sens_rep)):
            element = list()
            element.append(sens_rep[i])
            element.append(labels_rep[i])
            merged_rep.append(element)
        
        train_set,dev_set,test_set = torch.utils.data.random_split(merged_rep,[train_size,dev_size,test_size])
        x_train,x_dev,x_test,y_train,y_dev,y_test = [],[],[],[],[],[]
        for i in range(train_size):
            x_train.append(train_set[i][0])
            y_train.append(train_set[i][1])
        for i in range(dev_size):
            x_dev.append(dev_set[i][0])
            y_dev.append(dev_set[i][1])
        for i in range(test_size):
            x_test.append(test_set[i][0])
            y_test.append(test_set[i][1])
        
        # print(x_train[0])
        # print(y_train[0])
        # print(merged_rep.index([[1, 4, 1474, 1474, 284, 1474, 1474],13]))
        # print(merged_rep[4461])
        # print(len(merged_rep))
        # print(len(train_set))
        # print(len(dev_set))
        # print(len(test_set))

        # x_train,x_dev,x_test = torch.utils.data.random_split(sens_rep,[train_size,dev_size,test_size])
        # y_train,y_dev,y_test = torch.utils.data.random_split(labels_rep,[train_size,dev_size,test_size])

        # model = BOW_FFNN(voca_embs,int(config['hidden_size']),len(labels_index),bool(config['freeze']))