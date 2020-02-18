from data_preprocess import Preprocess

class Train:
    def __init__(self,tr_f,model,cfg_f,model_f):
        self.tr_f = tr_f
        self.model = model
        self.cfg_f = cfg_f
        self.model_f = model_f

    def train(self):
        read_config = open(self.cfg_f,'r')
        config = dict(read_config.read())

        prpr = Preprocess(self.tr_f)
        
        