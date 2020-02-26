from data_preprocess import Preprocess
import numpy as np
import torch
from bow_ffnn_pretrained import BOW_FFNN_PRE
from bow_ffnn_random import BOW_FFNN_RANDOM
from bilstm_ffnn_pretrained import BiLSTM_FFNN_PRE
from qc_dataset import QCDataset
from collate import qc_collate_fn_bilstm,qc_collate_fn_bow
from evaluation import get_accuracy_bow,get_accuracy_bilstm
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

        train_size = int(0.9*len(sens_rep))
        # dev_size = int(0.1*len(sens_rep))
        dev_size = len(sens_rep)-train_size

        merged_rep = list()
        for i in range(len(sens_rep)):
            element = list()
            element.append(sens_rep[i])
            element.append(labels_rep[i])
            merged_rep.append(element)
        
        train_set,dev_set = torch.utils.data.random_split(merged_rep,[train_size,dev_size])
        x_train,x_dev,y_train,y_dev = [],[],[],[]
        for i in range(train_size):
            x_train.append(train_set[i][0])
            y_train.append(train_set[i][1])
        for i in range(dev_size):
            x_dev.append(dev_set[i][0])
            y_dev.append(dev_set[i][1])

        

        xy_train = (x_train,y_train)
        xy_dev = (x_dev,y_dev)
        


        

        if(self.config["model"] == 'bow' and bool(self.config['from_pretrained'] == "True")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bow)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bow)

            model = BOW_FFNN_PRE(torch.FloatTensor(voca_embs),\
                int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bow' and bool(self.config['from_pretrained'] == "False")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bow)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bow)
            # I don't know why, but +1 works
            model = BOW_FFNN_RANDOM(len(vocabulary)+1,int(self.config['word_embedding_dim']),\
                int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bilstm' and bool(self.config['from_pretrained'] == "True")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bilstm)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bilstm)
            
            model = BiLSTM_FFNN_PRE(torch.FloatTensor(voca_embs),\
                int(self.config['bilstm_hidden_size']),int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bilstm' and bool(self.config['from_pretrained'] == "False")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bilstm)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bilstm)

            model = BOW_FFNN_RANDOM(len(vocabulary)+1,int(self.config['word_embedding_dim']),\
                int(self.config['bilstm_hidden_size']),int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))


        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),float(self.config['lr_param']),float(self.config['sgd_momentum']))

        model.train()
        early_stopping,best_accuracy = 0,0
        if(self.config['model']=='bilstm'):
            for epoch in range(int(self.config['epoch'])):
                batch = 1
                for data,label,length in loader_train:
                    optimizer.zero_grad()
                    y_pred = model(data,length)
                    loss = criterion(y_pred,torch.tensor(label)) 
                    batch += 1
                    loss.backward()
                    optimizer.step()
                    acc = get_accuracy_bilstm(model, loader_dev)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        early_stopping = 0
                        torch.save(model, self.config["path_model"])
                        print('Epoch {}, batch {}, best accuracy: {}'.format(epoch+1, batch, best_accuracy))
                    else :
                        early_stopping += 1
                    if early_stopping >= int(self.config["early_stopping"]):
                        print("Early stopping!")
                        break

            model = torch.load(self.config["path_model"])
            acc = get_accuracy_bilstm(model, loader_dev)
            print( "The accuray after training is : " , acc )

        if(self.config['model']=='bow'):
            for epoch in range(int(self.config['epoch'])):
                batch = 1
                for data,label in loader_train:
                    optimizer.zero_grad()
                    y_pred = model(data)
                    loss = criterion(y_pred,torch.tensor(label)) 
                    batch += 1
                    loss.backward()
                    optimizer.step()
                    acc = get_accuracy_bow(model, loader_dev)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        early_stopping = 0
                        torch.save(model, self.config["path_model"])
                        print('Epoch {}, batch {}, best accuracy: {}'.format(epoch+1, batch, best_accuracy))
                    else :
                        early_stopping += 1
                    if early_stopping >= int(self.config["early_stopping"]):
                        print("Early stopping!")
                        break

            model = torch.load(self.config["path_model"])
            acc = get_accuracy_bow(model, loader_dev)
            print( "The accuray after training is : " , acc )






