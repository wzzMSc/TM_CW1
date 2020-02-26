import torch
import numpy as np

def get_accuracy_bow(model,loader):
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

def get_accuracy_bilstm(model,loader):
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