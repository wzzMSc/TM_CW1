import torch
import numpy as np
import pandas as pd

def get_accuracy_bow(model,loader):
    y_preds = list()
    y_real = list()
    with torch.no_grad():
        for x, y in loader:
            y_preds.extend(model(x).argmax(dim=1).numpy().tolist())
            y_real.extend(y)
    return np.sum(np.array(y_preds)==y_real)/len(y_real),y_real,y_preds

def get_accuracy_bilstm(model,loader):
    y_preds = list()
    y_real = list()
    with torch.no_grad():
        for x, y, lengths in loader:
            y_preds.extend(model(x,lengths).argmax(dim=1).numpy().tolist())
            y_real.extend(y)
    return np.sum(np.array(y_preds)==y_real)/len(y_real),y_real,y_preds

def get_accuracy_test(model,model_type,x,y,lengths):
    with torch.no_grad():
        if model_type=='bow':
            y_preds = model(x).argmax(dim=1)
            return np.sum(y_preds.numpy()==y)/len(y)
        if model_type=='bilstm':
            y_preds = model(x,lengths).argmax(dim=1)
            return np.sum(y_preds.numpy()==y)/len(y)

def get_confusion_matrix(y_real,y_preds,size):
    mat = np.zeros( (size,size) )
    for i in range(len(y_real)):
        if y_real[i]==y_preds[i]:
            mat[y_real[i]-1][y_real[i]-1] += 1
        else:
            mat[y_real[i]-1][y_preds[i]-1] += 1
    return pd.DataFrame(mat)