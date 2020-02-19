from data_preprocess import Preprocess

class Train:
    def __init__(self,tr_f,model,cfg_f,model_f):
        self.tr_f = tr_f
        self.model = model
        self.cfg_f = cfg_f
        self.model_f = model_f

    def train(self):
        prpr = Preprocess(self.tr_f,self.cfg_f)
        # prpr.preprocess()
        labels,sentences,vocabulary,voca_embs,sens_rep,labels_rep = prpr.load_preprocessed()
        # for sen_rep in sens_rep:
        #     print(sen_rep)
        # for element in prpr.load_preprocessed():
        #     for i in range(5):
        #         print(element[i])
        