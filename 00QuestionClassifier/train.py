from data_preprocess import Preprocess

class Train:
    def __init__(self,tr_f,model,cfg_f,model_f):
        self.tr_f = tr_f
        self.model = model
        self.cfg_f = cfg_f
        self.model_f = model_f

    def train(self):

        
        print("ok")
        prpr = Preprocess(self.tr_f,self.cfg_f).preprocess()
        
        