from data_preprocess import Preprocess
import torch
from evaluation import get_accuracy_test

class Test:
    def __init__(self,config):
        self.config = config
    
    def test(self):
        prpr = Preprocess(self.config)
        prpr.preprocess(self.config['path_test'],"test")
        labels,sentences,vocabulary,voca_embs,sens_rep,labels_index,labels_rep = prpr.load_preprocessed()
        model = torch.load(self.config["path_model"])
        y = labels_rep
        lengths = list()
        for sen in sens_rep:
            lengths.append(len(sen))
        if(self.config["model"] == 'bow'):
            x = sens_rep
            print("Accuracy: {}".format(get_accuracy_test(model,"bow",x,y,False)))
        if(self.config["model"] == 'bilstm'):
            x = torch.nn.utils.rnn.pad_sequence(sens_rep,padding_value=0)
            print("Accuracy: {}".format(get_accuracy_test(model,"bow",x,y,lengths)))
