import torch

class FNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(FNN,self).__init__()
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inp):
        fc1 = self.fc1(inp)
        relu1 = self.relu1(fc1)
        fc2 = self.fc2(relu1)
        output =  self.softmax(fc2)
        return output