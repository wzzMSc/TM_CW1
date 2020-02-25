import torch
from torch import nn
from ffnn import FFNN

class BOW_FFNN_RANDOM(nn.Module):
    def __init__(self, voca_size, input_size, hidden_size, output_size, freeze = True):
        super(BOW_FFNN_RANDOM,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.freeze = freeze

        self.bow = nn.EmbeddingBag(voca_size,input_size,mode='mean')
        self.bow.weight.requires_grad = not self.freeze
        
        self.ffnn = FFNN(self.input_size,self.hidden_size,self.output_size)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input):
        batch = list()
        offsets = list()
        offset = 0
        for i in range(len(input)):
            batch += input[i]
            offsets.append(offset)
            offset += len(input[i])


        vec = self.bow(torch.tensor(batch),torch.tensor(offsets)) #CUDA
        ffnn = self.ffnn(vec)
        return self.log_softmax(ffnn)
