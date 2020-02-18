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

        preprocess = Preprocess(self.tr_f)
        labels,sentences = preprocess.preprocess()
        labels_index_dict = preprocess.get_label_index(labels)
        indexed_label = preprocess.get_indexed_labels(labels,labels_index_dict)
        words_count_list = preprocess.get_word_count(sentences)
        bow_list = preprocess.get_bow(sentences,config['word_embeddings_path'])
        
        