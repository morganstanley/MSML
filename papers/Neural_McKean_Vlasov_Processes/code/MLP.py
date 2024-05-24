import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_cond_inputs,
                 num_hid_layers,
                 num_hidden,
                 sigma=1,
                 act="tanh",
                 grad_sigma = True,
                 label_x = 0,
                 label_y = 0,
                 sigma_net = False,
                 act_last = False, 
                 **params
                ):
        super(MLP, self).__init__()
        self.W_0_hidden = 0
        self.sigma = nn.Parameter(torch.ones((1, num_outputs))*torch.tensor(sigma), requires_grad=grad_sigma)
        if num_outputs == 3:
            if sigma != 1:
                self.sigma = nn.Parameter(torch.tensor([sigma, sigma, sigma/2]).reshape(1, num_outputs),
                                      requires_grad=grad_sigma)
                
        if sigma_net:
            self.sigma_net = nn.Sequential(nn.Linear(1, num_hidden), nn.LeakyReLU(), nn.Linear(num_hidden, num_outputs), nn.Softplus())
        else:
            self.sigma_net = None
            
        self.num_cond_inputs = num_cond_inputs
        self.label_x = label_x
        self.label_y = label_y
        self.label_g = None
        self.label_f = None
        
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leakyrelu':nn.LeakyReLU}
        act_func  = activations[act]
        
        total_inputs = num_inputs + num_cond_inputs + label_x + 0
        mlp_modules = [nn.Linear(total_inputs, num_hidden), act_func()]
        for _ in range(num_hid_layers):
            mlp_modules += [nn.Linear(num_hidden, num_hidden), act_func()]
        mlp_modules += [nn.Linear(num_hidden, num_outputs, bias=True)] if act_last == False else [nn.Linear(num_hidden, num_outputs, bias=True), nn.Sigmoid()]
        self.f = nn.Sequential(*mlp_modules)
        
    def forward(self, x_t, y_t, t=None, particle_label_x=None, particle_label_y=None):
        y_t = None
        if self.num_cond_inputs == 0 or self.num_cond_inputs is None:
            inp = x_t
        elif self.label_x !=0:
            inp = torch.cat([x_t, t.repeat(x_t.shape[0],1), particle_label_x.reshape(particle_label_x.shape[0], 1)],1)
        else:
            inp = torch.cat([x_t, t.repeat(x_t.shape[0],1)],1)
        drift = self.f(inp)
        return drift
    
    def sigma_forward(self,t):
        if self.sigma_net != None:
            sigma = self.sigma_net(t.reshape(-1,1).to(next(self.sigma_net.parameters()).device))
            return sigma
        else:
            return self.sigma
    
    
    