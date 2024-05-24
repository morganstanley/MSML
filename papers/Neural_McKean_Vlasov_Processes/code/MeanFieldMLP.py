import warnings

import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from Glow import *

"""
TO-DO:
f function forward include labels
"""

class MF(nn.Module):
    def __init__(self,
                 num_g_inputs,
                 num_f_inputs,
                 num_g_outputs,
                 num_f_outputs,
                 g_num_hid_layers,
                 g_num_hidden,
                 f_num_hid_layers,
                 f_num_hidden,
                 W_0_hidden = 0,
                 W_0_init = "xavier_normal",
                 num_cond_inputs=None,
                 sigma = 1,
                 g_act='tanh',
                 f_act='tanh',
                 x_t_input = True,
                 res_input = True,
                 #label dimension
                 label_x = 0,
                 label_y = 0,
                 #label input
                 label_g = False,
                 label_f = False,
                 no_F = False, 
                 relative_meanfield = False,
                 grad_sigma = True,
                 W0_flow = False,
                 W0_flow_grid_init = False,
                 KL_init = False,
                 W0_flow_grid_space = 6,
                 l_num_hidden=64,
                 l_act='leakyrelu',
                 l_num_hid_layers=2,
                 linear_drift = False,
                 hetero_class = 1,
                 W0_Xt = False,
                 cond_on_g=True,
                 sigma_net=False,
                 act_last=False,
                 pe = 0,
                 **params
                ):
        
        def init_weight(m):
            if type(m) == nn.Linear:
                init.xavier_normal_(m.weight)
        
        if no_F == True:
            warnings.warn("Approximating without F, provide known F in training")
        super(MF, self).__init__()
        
        self.x_t_input = x_t_input
        self.res_input = res_input
        self.no_F = no_F
        self.cond_on_g = cond_on_g
        self.sigma = nn.Parameter(torch.ones((1, num_f_outputs))*sigma, requires_grad=grad_sigma)
        if num_f_outputs == 3:
            if sigma != 1:
                self.sigma = nn.Parameter(torch.tensor([sigma, sigma, sigma/2]).reshape(1, num_f_outputs),
                                      requires_grad=grad_sigma)
        if sigma_net:
            self.sigma_net = nn.Sequential(nn.Linear(1, f_num_hidden), nn.LeakyReLU(), nn.Linear(f_num_hidden, num_f_outputs), nn.Softplus())
        else:
            self.sigma_net = None
            
        self.W_0_hidden = W_0_hidden
        self.num_cond_inputs = num_cond_inputs
        self.relative_meanfield = relative_meanfield
        self.W0_flow = W0_flow
        self.linear_drift = linear_drift
        self.W0_Xt = W0_Xt
        
        # Label controls
        self.label_x = label_x
        self.label_y = label_y
        self.label_g = label_g
        self.label_f = label_f
        
        self.hetero_class = hetero_class
        
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leakyrelu':nn.LeakyReLU}
        g_act_func  = activations[g_act]
        f_act_func  = activations[f_act]
        l_act_func  = activations[l_act]
        
        # g function MLP
        if res_input:
            total_inputs = num_g_inputs + label_x + label_y
        else:
            total_inputs = num_g_inputs*2 + label_x + label_y
       
        if W_0_hidden != 0:
            total_inputs = num_g_inputs + (self.num_cond_inputs if self.num_cond_inputs != 0 else int(self.cond_on_g))
            if pe > 0:
                total_inputs = num_g_inputs + 2 * pe
            if label_g and self.hetero_class == 1:
                total_inputs = total_inputs + label_x + label_y
                
            if hetero_class == 1:
                self.W_0 = nn.Parameter(torch.zeros((W_0_hidden, num_g_inputs + label_y)))
            elif hetero_class > 1:
                self.W_0 = nn.Parameter(torch.zeros((hetero_class, W_0_hidden, num_g_inputs)))
            
            if W_0_init: 
                init.xavier_normal_(self.W_0, gain=1)
            # init W_0 differently during W0 flow case
            if self.W0_flow:
                if W0_flow_grid_init:
                    self.W_0.data=torch.linspace(-W0_flow_grid_space, 
                                                W0_flow_grid_space, 
                                                W_0_hidden).reshape(W_0_hidden, 1).repeat(1, num_g_inputs + label_y)
                else:
                    self.W_0.data=torch.normal(0,1, size=self.W_0.shape)
            # KL Expansion initialization
            if KL_init:
                ar = (torch.arange(self.W_0.shape[0]) + 1).reshape(self.W_0.shape[0],1)
                self.W_0.data = (torch.randn(self.W_0.shape) * torch.sin(0.5*np.sqrt(2)*np.pi*ar)/ar/np.pi).cumsum(0)

        g_modules = [nn.Linear(total_inputs, g_num_hidden), g_act_func()]
        for _ in range(g_num_hid_layers):
            g_modules += [nn.Linear(g_num_hidden, g_num_hidden), g_act_func()]
        g_modules += [nn.Linear(g_num_hidden, num_g_outputs)]
        self.g = nn.Sequential(*g_modules)
                
        # f function MLP
        total_inputs = num_f_inputs*2 + num_cond_inputs
        if self.linear_drift:
            total_inputs = num_f_inputs + num_cond_inputs
        if label_f:
                total_inputs = total_inputs + label_x + label_y
        
        f_modules = [nn.Linear(total_inputs, f_num_hidden), f_act_func()]
        for _ in range(f_num_hid_layers):
            f_modules += [nn.Linear(f_num_hidden, f_num_hidden), f_act_func()]
        f_modules += [nn.Linear(f_num_hidden, num_f_outputs)] if not act_last else [nn.Linear(f_num_hidden, num_f_outputs), nn.Sigmoid()]
        self.f = nn.Sequential(*f_modules)
        
        if self.W0_flow:
            # conditioning on t is required for W_0 diffusion 
            num_W0_flow_input = self.W_0.shape[-1]+1
            l_modules = [nn.Linear(num_W0_flow_input, l_num_hidden), l_act_func()]
            for _ in range(l_num_hid_layers):
                l_modules += [nn.Linear(l_num_hidden, l_num_hidden), l_act_func()]
            l_modules += [nn.Linear(l_num_hidden, self.W_0.shape[-1])]
            self.l = nn.Sequential(*l_modules)
            self.l.apply(init_weight)
            
        if self.W0_Xt:
            # for Radon-Nikodym derivative
            num_l_inputs = num_g_inputs + 1
            if pe > 0:
                num_l_inputs = num_g_inputs + 2 * pe
            l_modules = [nn.Linear(num_l_inputs, l_num_hidden), l_act_func()]
            for _ in range(l_num_hid_layers):
                l_modules += [nn.Linear(l_num_hidden, l_num_hidden), l_act_func()]
            l_modules += [nn.Linear(l_num_hidden, num_g_inputs)]
            self.l = nn.Sequential(*l_modules)
            self.l.apply(init_weight)

        if pe > 0:
            self.pe = PositionalEncodingLayer(pe)
        else:
            self.pe = None
            
        self.g.apply(init_weight)
        self.f.apply(init_weight)

    
    def forward(self, x_t, y_t, t=None, particle_label_x=None, particle_label_y=None):
        x_t = x_t.float()
        y_t = y_t.float() if y_t is not None else None
        t = t.float() if t is not None else None
        particle_label_x = particle_label_y.float() if particle_label_x is not None else None
        particle_label_y = particle_label_y.float() if particle_label_y is not None else None
        # Forward
        if self.W_0_hidden != 0 and self.res_input:
            # W0 forward
            if self.W0_flow and t > 0:
                # input W0 forward as W0 + L(W0, t)
                W0_flow_inp = torch.cat([self.W_0[:,self.label_y:], t.repeat(self.W_0.shape[0],1)], 1)
                y_t = self.W_0[:,self.label_y:] + self.l(W0_flow_inp)
            elif self.hetero_class > 1:
                y_t = self.W_0[particle_label_y.detach().cpu().tolist(),:,:].reshape(particle_label_y.shape[0]*self.W_0.shape[1], 
                                                             self.W_0.shape[-1])
            else:
                y_t = self.W_0[:,self.label_y:]
            res = x_t.repeat_interleave(y_t.shape[0], dim=0) - y_t.repeat(x_t.shape[0],1)
            
            if self.cond_on_g:
                if self.pe is not None:
                    t_p = self.pe(t)
                    res = torch.cat([res, t_p.repeat(res.shape[0],1)], 1)
                else:
                    res = torch.cat([res, t.repeat(res.shape[0],1)], 1)
            else:
                pass
            
        elif self.W_0_hidden == 0 and self.res_input:
            # Xt Architecture forward
            res = x_t.repeat_interleave(y_t.shape[0], dim=0) - y_t.repeat(x_t.shape[0],1)
        else:
            # Not using X_t - y_t
            res = torch.cat([x_t.repeat_interleave(y_t.shape[0], dim=0), y_t.repeat(x_t.shape[0],1)],1)
        
        # particle_label_y if label_y is not 0 (fow W_0 architecture only)
        try:
            particle_label_y = (self.W_0[:,0:self.label_y] if self.W_0[:,0:self.label_y].nelement() != 0 and \
                                                              self.hetero_class == 1 else particle_label_y)
        except:
            pass
        
        # Concate Labels
        if self.label_g and (self.hetero_class == 1 or self.W_0_hidden == 0):
            if particle_label_x is not None:
                res = torch.cat([res, particle_label_x.repeat_interleave(y_t.shape[0], dim=0).view(-1,1)],1)
            if particle_label_y is not None:
                res = torch.cat([res, particle_label_y.repeat_interleave(x_t.shape[0], dim=0).view(-1,1)],1)
        else: 
            pass
        
        if self.W0_Xt == False:
            Q_t = self.g(res).reshape(x_t.shape[0], y_t.shape[0], x_t.shape[1]).mean(1)
            
        elif self.W0_Xt == True:
            # MLP L denotes Radon-Nikodym Derivative
            if self.pe is not None:
                t_p = self.pe(t)
                l_inp = torch.cat([x_t.repeat_interleave(y_t.shape[0], dim=0), t_p.repeat(res.shape[0],1)],1)
            else:
                l_inp = torch.cat([x_t.repeat_interleave(y_t.shape[0], dim=0), t.repeat(res.shape[0],1)],1)

            Q_t = (self.g(res)*self.l(l_inp)).reshape(x_t.shape[0], y_t.shape[0], x_t.shape[1]).mean(1)
        elif self.no_F:
            return Q_t
        
        # Forward f
        # To-Do here, label input in f function
        if self.label_f:
            if particle_label_x is not None:
                inp = torch.cat([x_t, Q_t, t.repeat(x_t.shape[0],1), particle_label_x],1)
            if particle_label_y is not None:
                inp = torch.cat([inp, particle_label_y],1)
        elif self.linear_drift:
            inp = torch.cat([x_t, t.repeat(x_t.shape[0],1)],1)
        elif self.num_cond_inputs == 0 or self.num_cond_inputs is None:
            inp = torch.cat([x_t, Q_t,],1)
        else:
            if self.pe is not None:
                t_p = self.pe(t)
                inp = torch.cat([x_t, Q_t, t_p.repeat(x_t.shape[0],1)],1)
            else:
                inp = torch.cat([x_t, Q_t, t.repeat(x_t.shape[0],1)],1)
        drift = self.f(inp) + (Q_t if self.linear_drift else 0)

        return drift
        
    def sigma_forward(self, t):
        if self.sigma_net != None:
            #sigma = self.sigma_net(t.reshape(1,1))
            sigma = self.sigma_net(t.reshape(-1,1).to(next(self.sigma_net.parameters()).device))
            return sigma
        else:
            return self.sigma


class PositionalEncodingLayer(nn.Module):
    def __init__(self, L=20, device='cpu'):
        super(PositionalEncodingLayer, self).__init__()
        scale1 = 2**torch.arange(0, L)*math.pi
        scale2 = 2**torch.arange(0, L)*math.pi + math.pi
        self.scale = torch.stack((scale1,scale2),1).view(1,-1).to(device)

    def forward(self, x):
        xs = list(x.shape)
        vs = xs[:-1] + [-1]
        return torch.sin(x.unsqueeze(-1) @ self.scale.to(x.device)).view(*vs)
