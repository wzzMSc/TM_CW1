from data_preprocess import Preprocess
import numpy as np
import torch
from bow_ffnn_pretrained import BOW_FFNN_PRE
from bow_ffnn_random import BOW_FFNN_RANDOM
from bilstm_ffnn_pretrained import BiLSTM_FFNN_PRE
from qc_dataset import QCDataset
from torch.utils.data.dataloader import DataLoader

class Train:
    def __init__(self,config):
        self.config = config
        # print(config)

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

        
        # print(x_train[0])
        # print(y_train[0])
        # print(merged_rep.index([[1, 4, 1474, 1474, 284, 1474, 1474],13]))
        # print(merged_rep[4461])
        # print(len(merged_rep))
        # print(len(train_set))
        # print(len(dev_set))
        # print(len(test_set))

        xy_train = (x_train,y_train)
        xy_dev = (x_dev,y_dev)
        


        

        if(self.config["model"] == 'bow' and bool(self.config['from_pretrained'] == "True")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bow)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bow)

            model = BOW_FFNN_PRE(torch.FloatTensor(voca_embs),\
                int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bow' and bool(self.config['from_pretrained'] == "False")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bow)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bow)
            # I don't know why, but +1 works
            model = BOW_FFNN_RANDOM(len(vocabulary)+1,int(self.config['word_embedding_dim']),\
                int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bilstm' and bool(self.config['from_pretrained'] == "True")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bilstm)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bilstm)
            
            model = BiLSTM_FFNN_PRE(torch.FloatTensor(voca_embs),\
                int(self.config['bilstm_hidden_size']),int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bilstm' and bool(self.config['from_pretrained'] == "False")):
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bilstm)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=self.qc_collate_fn_bilstm)

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
                    print('Epoch {}, batch {}, best accuracy: {}'.format(epoch+1, batch, best_accuracy))
                    batch += 1
                    loss.backward()
                    optimizer.step()
                    acc = self.get_accuracy_bilstm(model, loader_dev)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        early_stopping = 0
                        torch.save(model, self.config["path_model"])
                    else :
                        early_stopping += 1
                    if early_stopping >= int(self.config["early_stopping"]):
                        print("Early stopping!")
                        break

            model = torch.load(self.config["path_model"])
            acc = self.get_accuracy_bilstm(model, loader_dev)
            print( "The accuray after training is : " , acc )

        if(self.config['model']=='bow'):
            for epoch in range(int(self.config['epoch'])):
                batch = 1
                for data,label in loader_train:
                    optimizer.zero_grad()
                    y_pred = model(data)
                    loss = criterion(y_pred,torch.tensor(label)) 
                    print('Epoch {}, batch {}, best accuracy: {}'.format(epoch+1, batch, best_accuracy))
                    batch += 1
                    loss.backward()
                    optimizer.step()
                    acc = self.get_accuracy_bow(model, loader_dev)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        early_stopping = 0
                        torch.save(model, self.config["path_model"])
                    else :
                        early_stopping += 1
                    if early_stopping >= int(self.config["early_stopping"]):
                        print("Early stopping!")
                        break

            model = torch.load(self.config["path_model"])
            acc = self.get_accuracy_bow(model, loader_dev)
            print( "The accuray after training is : " , acc )



    def qc_collate_fn_bow(self,QCDataset):
        data,label = [],[]
        for dataset in QCDataset:
            data.append(dataset[0])
            label.append(dataset[1])
        return data,label

    def qc_collate_fn_bilstm(self,QCDataset):
        length,data,label = [],[],[]
        for dataset in QCDataset:
            data.append(dataset[0])
            label.append(dataset[1])
            length.append(len(dataset[0]))
        data = torch.nn.utils.rnn.pad_sequence(data,padding_value=0)
        return data,label,length

    def get_accuracy_bow(self,model,loader):
        count = 0
        length =0
        with torch.no_grad():
            for x, y in loader:
                y_preds = model(x).argmax(dim=1)
                count += np.sum(y_preds.numpy() == y) 
                length += len(y)
            # compute the accuracy
        acc = count/length
        return acc
    
    def get_accuracy_bilstm(self,model,loader):
        count = 0
        length =0
        with torch.no_grad():
            for x, y, lengths in loader:
                y_preds = model(x,lengths).argmax(dim=1)
                count += np.sum(y_preds.numpy() == y) 
                length += len(y)
            # compute the accuracy
        acc = count/length
        return acc
