from data_preprocess import Preprocess
import numpy as np
import torch
from bow_ffnn_pretrained import BOW_FFNN_PRE
from bow_ffnn_random import BOW_FFNN_RANDOM
from bilstm_ffnn_pretrained import BiLSTM_FFNN_PRE
from bilstm_ffnn_random import BiLSTM_FFNN_RANDOM
from bow_bilstm_pretrained import BOW_BiLSTM_PRE
from bow_bilstm_random import BOW_BiLSTM_RANDOM
from qc_dataset import QCDataset
from collate import qc_collate_fn_bilstm,qc_collate_fn_bow
from evaluation import get_accuracy_bow,get_accuracy_bilstm,get_confusion_matrix,get_micro_f1,get_macro_f1
from torch.utils.data.dataloader import DataLoader

class Train:
    def __init__(self,config):
        self.config = config

    def train(self):

        # Preprocess the data
        prpr = Preprocess(self.config)
        # Comment this line below to skip the data preprocessing if you have preprocessed the data before
        prpr.preprocess(self.config['path_data'],"train")
        # Get all essential data preprocessing results
        labels,sentences,vocabulary,voca_embs,sens_rep,labels_index,labels_rep = prpr.load_preprocessed()

        # Train:Dev=9:1
        train_size = int(0.9*len(sens_rep))
        dev_size = len(sens_rep)-train_size
        # Combine the representation of sentences and labels for data split
        merged_rep = list()
        for i in range(len(sens_rep)):
            element = list()
            element.append(sens_rep[i])
            element.append(labels_rep[i])
            merged_rep.append(element)
        # Set random seed to make the result repeatable
        torch.manual_seed(0)
        # Randomly split the data
        train_set,dev_set = torch.utils.data.random_split(merged_rep,[train_size,dev_size])
        # Unpack the splitted data
        x_train,x_dev,y_train,y_dev = [],[],[],[]
        for i in range(train_size):
            x_train.append(train_set[i][0])
            y_train.append(train_set[i][1])
        for i in range(dev_size):
            x_dev.append(dev_set[i][0])
            y_dev.append(dev_set[i][1])   
        # For objects of type QCDataset
        xy_train = (x_train,y_train)
        xy_dev = (x_dev,y_dev)
        
        # Set up the dataloaders and models
        if self.config["model"]=='bow':
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bow)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bow)
        if self.config["model"]=='bilstm' or self.config["model"]=='bow_bilstm':
            qc_train = QCDataset(xy_train)
            loader_train = DataLoader(qc_train,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bilstm)
            qc_dev = QCDataset(xy_dev)
            loader_dev = DataLoader(qc_dev,batch_size=int(self.config["batch_size"]),collate_fn=qc_collate_fn_bilstm)

        
        if(self.config["model"] == 'bow' and bool(self.config['from_pretrained'] == "True")):
            model = BOW_FFNN_PRE(torch.FloatTensor(voca_embs),\
                int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bow' and bool(self.config['from_pretrained'] == "False")):
            model = BOW_FFNN_RANDOM(len(vocabulary)+1,int(self.config['word_embedding_dim']),\
                int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bilstm' and bool(self.config['from_pretrained'] == "True")):
            model = BiLSTM_FFNN_PRE(torch.FloatTensor(voca_embs),\
                int(self.config['bilstm_hidden_size']),int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bilstm' and bool(self.config['from_pretrained'] == "False")):
            model = BiLSTM_FFNN_RANDOM(len(vocabulary)+1,int(self.config['word_embedding_dim']),\
                int(self.config['bilstm_hidden_size']),int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))
        
        if(self.config["model"] == 'bow_bilstm' and bool(self.config['from_pretrained'] == "True")):
            model = BOW_BiLSTM_PRE(torch.FloatTensor(voca_embs),\
                int(self.config['bilstm_hidden_size']),int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        if(self.config["model"] == 'bow_bilstm' and bool(self.config['from_pretrained'] == "False")):
            model = BOW_BiLSTM_RANDOM(len(vocabulary)+1,int(self.config['word_embedding_dim']),\
                int(self.config['bilstm_hidden_size']),int(self.config['hidden_size']),len(labels_index),bool(self.config['freeze']=="True"))

        # Set the criterion as CrossEntropyLoss
        criterion = torch.nn.CrossEntropyLoss()
        # Use SGD optimizer, set learning rate and momentum
        optimizer = torch.optim.SGD(model.parameters(),float(self.config['lr_param']),float(self.config['sgd_momentum']))
        # Start to train the FFNN
        model.train()
        early_stopping,best_accuracy = 0,0
        if(self.config['model']=='bilstm' or self.config['model']=='bow_bilstm'):
            for epoch in range(int(self.config['epoch'])):
                batch = 1
                # Use dataloader to load data in batch fashion
                for data,label,length in loader_train:
                    optimizer.zero_grad()
                    y_pred = model(data,length)
                    loss = criterion(y_pred,torch.tensor(label)) 
                    batch += 1
                    loss.backward()
                    optimizer.step()
                    # Calculate the accuracy of model using development set
                    acc,_,_ = get_accuracy_bilstm(model, loader_dev)
                    # If model gets better
                    if acc > best_accuracy:
                        # Refresh the best accuracy record
                        best_accuracy = acc
                        # Reset early stopping counter to zero
                        early_stopping = 0
                        # Save current model
                        torch.save(model, self.config["path_model"])
                        # Print the current result
                        print('Epoch {}, batch {}, best accuracy: {}'.format(epoch+1, batch, best_accuracy))
                    else :
                        # If the model doesn't perform better, increase early stopping counter by one
                        early_stopping += 1
                    if early_stopping >= int(self.config["early_stopping"]):
                        # If the early stopping counter exceeds the threshold, stop training the models
                        print("Early stopping!")
                        break
            # Load the best model
            model = torch.load(self.config["path_model"])
            # Get the accuracy of model on dev set, get the actual and predicted value of labels
            acc,y_real,y_pre = get_accuracy_bilstm(model, loader_dev)
            # Get the confusion matrix of results on dev set
            conf_mat = get_confusion_matrix(y_real,y_pre,len(labels_index))
            # Compute the micro and marco F1
            micro_f1 = get_micro_f1(conf_mat)
            macro_f1 = get_macro_f1(conf_mat)
            print( "The accuray after training is : " , acc )
            print("Confusion Matrix:\n",conf_mat)
            print("Micro F1: ",micro_f1)
            print("Macro F1: ",macro_f1)
            output = open(self.config["path_eval_result"],'w')
            print("{0:<15}\t{1:<15}\t{2}".format("Actual","Prediction","Correct?"),file = output)
            for i,j in zip(y_real,y_pre):
                real = list(labels_index.keys())[list(labels_index.values()).index(i)]
                pre = list(labels_index.keys())[list(labels_index.values()).index(j)]
                if i==j:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"True"),file = output)
                else:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"False"),file = output)
            print(
                "The accuray after training is: ",acc,
                '\n',
                "Confusion Matrix:\n",conf_mat,
                '\n',
                "Micro F1: ",micro_f1,
                '\n',
                "Macro F1: ",macro_f1,
                file = output
            )
            output.close()

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
                    acc,_,_ = get_accuracy_bow(model, loader_dev)
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
            acc,y_real,y_pre = get_accuracy_bow(model, loader_dev)
            conf_mat = get_confusion_matrix(y_real,y_pre,len(labels_index))
            micro_f1 = get_micro_f1(conf_mat)
            macro_f1 = get_macro_f1(conf_mat)
            print( "The accuray after training is : " , acc )
            print("Confusion Matrix:\n",conf_mat)
            print("Micro F1: ",micro_f1)
            print("Macro F1: ",macro_f1)
            output = open(self.config["path_eval_result"],'w')
            print("{0:<15}\t{1:<15}\t{2}".format("Actual","Prediction","Correct?"),file = output)
            for i,j in zip(y_real,y_pre):
                real = list(labels_index.keys())[list(labels_index.values()).index(i)]
                pre = list(labels_index.keys())[list(labels_index.values()).index(j)]
                if i==j:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"True"),file = output)
                else:
                    print("{0:<15}\t{1:<15}\t{2}".format(real,pre,"False"),file = output)
            print(
                "The accuray after training is: ",acc,
                '\n',
                "Confusion Matrix:\n",conf_mat,
                '\n',
                "Micro F1: ",micro_f1,
                '\n',
                "Macro F1: ",macro_f1,
                file = output
            )
            output.close()
