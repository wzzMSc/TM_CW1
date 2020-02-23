import torch
from torch import nn
from ffnn import FFNN

class BiLSTM_FFNN_PRE(nn.Module):
    def __init__(self, embeddings, bilstm_hidden_size, ffnn_hidden_size, ffnn_output_size, freeze = True):
        super(BiLSTM_FFNN_PRE, self).__init__()
        _, self.input_size = embeddings.size()
        self.bilstm_hidden_size = bilstm_hidden_size
        self.fnn_hidden_size = ffnn_hidden_size
        self.fnn_output_size = ffnn_output_size
        self.freeze = freeze

        self.embeddingLayer = nn.Embedding.from_pretrained(embeddings, self.freeze)
        
        self.bilstm = nn.LSTM(self.input_size,self.bilstm_hidden_size, bidirectional=True)
        self.fnn = FFNN(self.bilstm_hidden_size*2,self.fnn_hidden_size, self.fnn_output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data, lengths):
        sequence_embed = self.embeddingLayer(data)
        pack = torch.nn.utils.rnn.pack_padded_sequence(
            sequence_embed,
            lengths, enforce_sorted=False
        )
        packed_output,_ = self.bilstm(pack)
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        vec = output.mean(dim=0)
        fnn = self.fnn(vec)
        return self.log_softmax(fnn)