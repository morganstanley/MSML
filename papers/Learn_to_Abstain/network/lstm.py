import torch
from torch import nn

class LSTMSeq(nn.Module):
    def __init__(self,input_dim=1, emb_size=512, num_layer=2, num_class=2, dropout=0.05,device='cuda', **kwargs):
        super(LSTMSeq, self).__init__()
        self.num_class = num_class
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.emb_size = emb_size
        self.dropout = dropout
        self.device = device
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = emb_size, num_layers = num_layer,dropout = dropout,batch_first = True)
        self.fc3 = nn.Linear(emb_size,num_class)

    def forward(self, x, *args, **kwargs):
        self.h_0 = torch.tensor(torch.zeros(self.num_layer,x.size(0),self.emb_size),requires_grad=True).to(self.device)
        self.c_0 = torch.tensor(torch.zeros(self.num_layer,x.size(0),self.emb_size),requires_grad=True).to(self.device)
        lstm_out, (self.h_0, self.c_0) = self.lstm(x,(self.h_0, self.c_0))
        out = self.fc3(lstm_out)
        return out

class LSTMSeqSelectiveNet(nn.Module):
        def __init__(self,input_dim=1, emb_size=128, num_layer=2, num_class=2, dropout=0.05, device='cuda', **kwargs):
            super(LSTMSeqSelectiveNet, self).__init__()
            self.num_class = num_class
            self.num_layer = num_layer
            self.input_dim =input_dim
            self.emb_size = emb_size
            self.dropout = dropout
            self.device = device
            self.lstm = nn.LSTM(input_size = input_dim, hidden_size = emb_size,num_layers = num_layer,dropout = dropout,batch_first = True)
            self.fc = nn.Linear(emb_size, num_class)
            self.selfc = nn.Linear(emb_size, 1)
            self.augfc = nn.Linear(emb_size, num_class)
        
        def forward(self,x, *args, **kwargs):

            h_0 = torch.tensor(torch.zeros(self.num_layer,x.size(0), self.emb_size), requires_grad=True).to(self.device)
            c_0 = torch.tensor(torch.zeros(self.num_layer,x.size(0), self.emb_size), requires_grad=True).to(self.device)

            lstm_out, (h_out, c_out) = self.lstm(x,(h_0, c_0))
            cls_out = self.fc(lstm_out)
            sl_out  = torch.sigmoid(self.selfc(lstm_out))
            aug_out = self.augfc(lstm_out)
            return cls_out, sl_out, aug_out