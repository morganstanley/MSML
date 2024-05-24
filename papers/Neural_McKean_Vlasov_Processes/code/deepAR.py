import torch
from torch import nn


class deepAR_net(nn.Module):
    def __init__(self,
                 rnn,
                 num_rnn_inputs,
                 num_rnn_layers,
                 num_rnn_hidden,
                 num_dense_layers,
                 num_dense_hidden,
                 num_outputs,
                 num_cond_inputs=None,
                 bidirectional = False,
                 act='leakyrelu',
                 sigma=1,
                 **params
                ):
        super(deepAR_net, self).__init__()
        if sigma == False:
            pass
        else:
            self.sigma = nn.Parameter(torch.ones((1, num_outputs))*sigma)
        rnn_units = {'lstm':nn.LSTM, 'gru':nn.GRU, 'rnn':nn.RNN}
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leakyrelu':nn.LeakyReLU}
        self.rnn_hidden = num_rnn_hidden
        tot_inputs = num_rnn_inputs
        
        rnn_unit = rnn_units[rnn]
        self.rnn_net = rnn_unit(input_size=num_rnn_inputs, 
                                hidden_size=num_rnn_hidden, 
                                num_layers=1 + num_rnn_layers,
                                batch_first = True,
                                bidirectional = bidirectional
                               )
        
        mu_act = activations[act]
        mu_modules = [nn.Linear(num_rnn_hidden, num_dense_hidden), mu_act()]
            
        for _ in range(num_dense_layers):
            mu_modules += [nn.Linear(num_dense_hidden, num_dense_hidden), mu_act()]
            if _ == num_dense_layers-1:
                mu_modules += [nn.Linear(num_dense_hidden, num_outputs)]
        
        sig_act = nn.Softplus
        sig_modules = [nn.Linear(num_rnn_hidden, num_dense_hidden), sig_act()]
        for _ in range(num_dense_layers):
            sig_modules += [nn.Linear(num_dense_hidden, num_dense_hidden), sig_act()]
            if _ == num_dense_layers-1:
                sig_modules += [nn.Linear(num_dense_hidden, num_outputs), sig_act()]
                

        if num_dense_layers == 0:
            mu_modules = [nn.Linear(num_rnn_hidden, num_outputs)]
            sig_modules= [nn.Linear(num_rnn_hidden, num_outputs), sig_act()]
            
        self.dense_mu = nn.Sequential(*mu_modules)
        self.dense_sig= nn.Sequential(*sig_modules)
    
    def forward(self, x, t=None, hidden=None):
        if t is not None:
            rnn_in = torch.cat([x,t], -1)
        else:
            rnn_in = x
        rnn_out, hidden = self.rnn_net(rnn_in, hidden)
        rnn_out = rnn_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        dense_out = self.dense_mu(rnn_out)
        sig_out   = self.dense_sig(rnn_out)
        return dense_out, sig_out, hidden
    