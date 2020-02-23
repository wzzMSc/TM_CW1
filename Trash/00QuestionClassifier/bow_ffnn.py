import torch
from torch import nn
from ffnn import FFNN

class BOW_FFNN(nn.Module):
    def __init__(self, embeddings, hidden_size, output_size, freeze = True):
        super(BOW_FFNN,self).__init__()
        self.voca_size,self.input_size = embeddings.size()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.freeze = freeze
        self.bow = nn.EmbeddingBag.from_pretrained(embeddings,self.freeze,mode='mean')
        self.ffnn = FFNN(self.input_size,self.hidden_size,self.output_size)

    def forward(self,input):
        return nn.functional.log_softmax(self.ffnn(self.bow(input,torch.LongTensor([0]))))