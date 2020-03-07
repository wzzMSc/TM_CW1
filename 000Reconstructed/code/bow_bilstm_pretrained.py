from torch import nn
import torch
from ffnn import FFNN

class BOW_BILSTM_PRE(nn.Module):

    def __init__(self, embeddings, bilstm_hidden_size, ffnn_hidden_size, ffnn_output_size, freeze = True):
        super(BOW_BILSTM_PRE, self).__init__()
        self.freeze = freeze
        _, self.input_size = embeddings.size()
        self.bilstm_hidden_size = bilstm_hidden_size
        self.ffnn_hidden_size = ffnn_hidden_size
        self.ffnn_output_size = ffnn_output_size

        self.emb = nn.Embedding.from_pretrained(embeddings, self.freeze)
        self.bow = nn.EmbeddingBag.\
                from_pretrained(
                    embeddings,
                    freeze=self.freeze,
                    mode='mean'
                )
        self.lstm = nn.LSTM(self.input_size,self.bilstm_hidden_size, bidirectional=True)
        self.ffnn = FFNN(self.bilstm_hidden_size*2,self.ffnn_hidden_size, self.ffnn_output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, lengths):
        input = input.transpose(0,1)
        batch = list()
        offsets = list()
        offset = 0
        for i in range(input.size()[0]):
            batch += input[i][:lengths[i]]
            offsets.append(offset)
            offset += lengths[i]

        vec = self.bow(torch.tensor(batch), torch.tensor(offsets))
        lstm_out, _ = self.lstm(vec.view(1, -1, self.input_size))

        ffnn = self.ffnn(lstm_out.view(-1, self.bilstm_hidden_size*2))
        return self.log_softmax(ffnn)
