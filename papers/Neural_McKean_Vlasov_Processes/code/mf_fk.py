import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

from nets_fk import SMLP
import numpy as np

import pytorch_lightning as pl

from torchvision.utils import make_grid
from MeanFieldMLP import MF
from MLP import MLP
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

D_IDX = -1
T_IDX = -2
N_IDX =  0

class FKModule(pl.LightningModule):
    def __init__(self, savepath, plot_savepath, net_savepath, 
                 test_data, val_data, device,
                 d, mu, g, T = 0.1, dt = 0.005, 
                 N = 10, lr_mu = 1e-2, noise='diag', 
                 ):
        # T = 0.1
        # dt = 0.005
        super().__init__()

        # dimension
        self.d = d

        # N expectation
        self.N = N

        # boundary and drift functions
        self.mu = mu
        self.pmu = torch.zeros(d, requires_grad=True, device=device)
        self.ps  = torch.eye(d, requires_grad=True, device=device)
        

        mn = torch.distributions.multivariate_normal.MultivariateNormal(self.pmu, self.ps.abs())
        g = lambda x: mn.log_prob(x).exp()

        self.g  = g

        # integration time
        t   = torch.linspace(0, T, int(T/dt)).to(device)
        self.register_buffer('t', t)

        # N of expectation x 1 x T x d
        dWt = torch.randn(N, 1, self.t.shape[0], d) * np.sqrt(dt)
        self.register_buffer('dWt', dWt)

        # set the sigma

        if noise == 'diag':
            self.sigma = torch.ones(d, requires_grad=True, device=device)
        elif noise == 'full':
            self.s  = torch.zeros((self.d,self.d), device=device)
            self.si = torch.tril_indices(self.d, self.d)
            vals = torch.eye(self.d)[self.si[0,:], self.si[1,:]]
            self.sigma = torch.tensor(vals, requires_grad=True, device=device)

        self.lr_mu = lr_mu
        
        self.plot_savepath = plot_savepath
        self.test_savepath = savepath
        self.net_savepath  = net_savepath
        
        self.val_data = val_data
        self.val_loss = 0
        self.val_samples = []
        
        self.test_data = test_data
        self.test_loss     = 0
        self.test_samples  = []

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x = x.reshape(-1, self.d)

        #if ( ( self.current_epoch + 1 ) % 200 ) == 0 and batch_idx == 0:
        #    self.plot_p(x)

        return -self.log_p_E(x)

    def plot_p(self, x):
        K = 30

        # plot the vector fields 
        x,y = torch.meshgrid(torch.linspace(x[:,0].min(),x[:,0].max(),K).type_as(x), torch.linspace(x[:,1].min(),x[:,1].max(),K).type_as(x))
        X_ = torch.stack((x,y),-1).reshape(-1,2)
        P = torch.zeros(X_.shape[0])

        for idx in range(X_.shape[0]):
            P[idx] = self.p(X_[idx,:].unsqueeze(0)).detach()

        print(P.max())
        print(P.min())

        P[P > P.mean() + 2 * P.std()] = 0 

        plt.imshow(P.view(K,K).cpu().detach(), cmap='turbo')
        plt.colorbar()
        plt.savefig('px.png')
        plt.close('all')

    def validation_step(self, batch, batch_idx):
        device = self.pmu.device
        x_data = batch[0]
        N_samp = x_data.shape[0]
        samples, xinit = self.sample(x_data, N=N_samp)
        samples = samples.detach().cpu()
        self.val_samples.append(samples)
        sampled = torch.cat(self.val_samples, 0).detach().cpu()
        
        xinit = xinit.detach().cpu()
        x0 = x_data.detach().cpu()

        if x_data.shape[-1] == 2:
            fig = plt.figure(figsize=(4,4), dpi=200)
            # plot the points
            plt.scatter(self.val_data[...,0], self.val_data[...,1], alpha=0.1, label='Real')
            plt.scatter(sampled[:,0], sampled[:,1], alpha=0.1, label=r'$X_T$')
            #plt.scatter(xinit[:,0], xinit[:,1], alpha=0.1, label=r'$X_0$')
            plt.legend()
            
            plt.savefig(self.plot_savepath + 'samples_val_{}.pdf'.format(self.current_epoch))
            plt.close('all')

            K = 10
            # plot the vector fields 
            xmin = self.val_data[:,0].min().detach().cpu().numpy()
            xmax = self.val_data[:,0].max().detach().cpu().numpy()
            ymin = self.val_data[:,1].min().detach().cpu().numpy()
            ymax = self.val_data[:,1].max().detach().cpu().numpy()
            
            x,y = np.meshgrid(np.linspace(xmin*1.5,xmax*1.5,K), np.linspace(ymin*1.5,ymax*1.5,K))
            
            X_ = torch.stack((torch.from_numpy(x),torch.from_numpy(y)),-1).reshape(-1,2).float().to(device)
            fig = plt.figure(figsize=(4,4), dpi=200)
            sns.kdeplot(x = sampled[...,0], y = sampled[...,1], shade=True, cmap = "Reds")
            
            drift = self.mu(X_, X_, self.t[-1]).detach().cpu().numpy()
            drift_x = drift[:,0].reshape(x.shape[0], x.shape[1])
            drift_y = drift[:,1].reshape(y.shape[0], y.shape[1])
            
            drift_strength = (drift_x*drift_y)
            drift_strength_minmax = ((drift_strength - drift_strength.min())/(drift_strength.max() - drift_strength.min()))
            plt.streamplot(x, y, drift_x, drift_y,
                            color="grey", density=2, arrowsize=0.5,
                            linewidth=2*drift_strength_minmax,
                            arrowstyle="-|>")
            
            plt.xlim(xmin*1.4, xmax*1.4)
            plt.ylim(ymin*1.4, ymax*1.4)

            plt.savefig(self.plot_savepath + 'gradient_{}.pdf'.format(self.current_epoch))
            plt.close('all')
            
        if x_data.shape[-1] == 3:
            fig = plt.figure(figsize=(8,4))
            x, y, z = self.val_data.detach().cpu().numpy().T
            ax_temp = fig.add_subplot(1, 2, 1, projection='3d')
            ax_temp.scatter(x, y, z)
            ax_temp.view_init(azim=-60, elev=9)
            x, y, z = sampled.T
            ax_temp = fig.add_subplot(1, 2, 2, projection='3d')
            ax_temp.scatter(x, y, z)
            ax_temp.view_init(azim=-60, elev=9)
            
            plt.savefig(self.plot_savepath + 'samples_val_{}.pdf'.format(self.current_epoch))
            
        if x_data.shape[-1] <= 100 and x_data.shape[-1] >= 4:
            fig = plt.figure(figsize=(4,4), dpi=200)
            # plot the points
            plt.scatter(self.val_data[...,0], self.val_data[...,1], alpha=0.1, label='Real')
            plt.scatter(sampled[:,0], sampled[:,1], alpha=0.1, label=r'$X_T$')
            #plt.scatter(xinit[:,0], xinit[:,1], alpha=0.1, label=r'$X_0$')
            plt.legend()
            
            plt.savefig(self.plot_savepath + 'samples_val_{}.pdf'.format(self.current_epoch))
            plt.close('all')
            
    def validation_epoch_end(self, outputs=None):
        self.val_samples = []
                
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        x_data = batch[0]
        x_data = x_data.reshape(-1, self.d)
        test_loss = -self.log_p_E(x_data).detach()
        self.test_loss  += test_loss.cpu()
        N_samp = x_data.shape[0]

        samples, _ = self.sample(x_data, N=N_samp)
        self.test_samples.append(samples)
        
        sampled = torch.cat(self.test_samples, 0).detach().cpu()
        device = self.pmu.device
        if x_data.shape[-1] == 2:
            # plot the points
            plt.figure(figsize=(4,4))
            plt.scatter(self.test_data[:,0], self.test_data[:,1], alpha=0.1, label='Real')
            if isinstance(self.mu, MF):
                if self.mu.W_0_hidden > 0:
                    label = r"$W_0$"
                else:
                    label = r"$X_t$"
            else:
                label = "MLP"
            plt.scatter(sampled[:,0], sampled[:,1], alpha=0.1, label="Generated")
            #plt.scatter(xinit[:,0], xinit[:,1], alpha=0.1, label=r'$X_0$')
            plt.legend()
            
            plt.savefig(self.test_savepath + 'Test_samples_val.pdf')
            plt.close('all')

            K = 20
            xmin = self.test_data[:,0].min().detach().cpu().numpy()
            xmax = self.test_data[:,0].max().detach().cpu().numpy()
            ymin = self.test_data[:,1].min().detach().cpu().numpy()
            ymax = self.test_data[:,1].max().detach().cpu().numpy()
            # plot the vector fields 
            x,y = np.meshgrid(np.linspace(xmin*2,xmax*2,K), np.linspace(ymin*2,ymax*2,K))
            X_ = torch.stack((torch.from_numpy(x),torch.from_numpy(y)),-1).reshape(-1,2).float().to(device)
            plt.figure(figsize=(4,4))
            sns.kdeplot(x=sampled[...,0], y=sampled[...,1], shade=True, cmap = "Reds")
            
            drift = self.mu(X_, X_, self.t[-1]).detach().cpu().numpy()
            drift_x = drift[:,0].reshape(x.shape[0], x.shape[1])
            drift_y = drift[:,1].reshape(y.shape[0], y.shape[1])
            
            drift_strength = (drift_x*drift_y)
            drift_strength_minmax = ((drift_strength - drift_strength.min())/(drift_strength.max() - drift_strength.min()))
            
            plt.streamplot(x, y, drift_x, drift_y,
                           color="grey", density=2, arrowsize=0.5,
                           linewidth=1.5*drift_strength_minmax,
                           arrowstyle="-|>")
            
            plt.xlim(xmin*1.8, xmax*1.8)
            plt.ylim(ymin*1.8, ymax*1.8)
            
            plt.savefig(self.test_savepath + 'Test_gradient.pdf')
            plt.close('all')
            
        if x_data.shape[-1] == 3:
            fig = plt.figure(figsize=(8,4))
            x, y, z = self.test_data.detach().cpu().numpy().T
            ax_temp = fig.add_subplot(1, 2, 1, projection='3d')
            ax_temp.scatter(x, y, z)
            ax_temp.view_init(azim=-60, elev=9)
            x, y, z = sampled.T
            ax_temp = fig.add_subplot(1, 2, 2, projection='3d')
            ax_temp.scatter(x, y, z)
            ax_temp.view_init(azim=-60, elev=9)
            
            plt.savefig(self.plot_savepath + 'samples_val_{}.pdf'.format(self.current_epoch))
        
        if x_data.shape[-1] <= 100 and x_data.shape[-1] >= 4:
            fig = plt.figure(figsize=(4,4), dpi=200)
            # plot the points
            plt.scatter(self.test_data[...,0], self.test_data[...,1], alpha=0.1, label='Real')
            plt.scatter(sampled[:,0], sampled[:,1], alpha=0.1, label="Generated")
            #plt.scatter(xinit[:,0], xinit[:,1], alpha=0.1, label=r'$X_0$')
            plt.legend()
            
            plt.savefig(self.net_savepath + 'Test_samples_val.pdf'.format(self.current_epoch))
            plt.close('all')
                
        test_stats = (sampled, self.test_loss)
        with open(os.path.join(self.test_savepath,'test_stats.pkl'), 'wb') as f:
            pickle.dump(test_stats, f)
            f.close()

    def log_p_E(self, x):
        x = x.float()
        # x shape is batch size x d
        dt = self.t[1] - self.t[0]

        if self.sigma.shape[0] == self.d:
            sigma = self.sigma.abs()
            dWt = (sigma * torch.randn(self.N, 1, self.t.shape[0], self.d).type_as(x) * torch.sqrt(dt)).repeat(1,x.shape[0],1,1) # TODO: check difference
        else:
            s = torch.zeros((self.d,self.d), device=self.sigma.device)
            s[self.si[0,:], self.si[1,:]] = self.sigma.abs()
            sigma = s
            sigma_i = torch.cholesky_inverse(s) 
            dWt = (torch.randn(self.N, x.shape[0], self.t.shape[0], self.d).type_as(x) * torch.sqrt(dt) @ sigma) # TODO: check difference
            
        Wt  = dWt.cumsum(-2) + x.unsqueeze(1) # compute brownian path up to time T
        Wt[:,:,0,:] = x
        Wt  =  Wt.reshape(-1,  Wt.shape[2],  Wt.shape[3])
        dWt = dWt.reshape(-1, dWt.shape[2], dWt.shape[3])
        
        like = 0 
        for t_idx in range(Wt.shape[1]):
            mu  = self.mu(Wt[:,t_idx], Wt[:,t_idx], t=self.t[t_idx])
            tr_dmu = self.trace_grad(mu, Wt[:,t_idx],t=self.t[t_idx], N=self.N)  # trace of jacobian
            v = tr_dmu
            if self.sigma.shape[0] == self.d:
                a = (mu * dWt[:,t_idx] / sigma).sum(-1)                # first term in girsanov
                b = - 1/2 * dt * ( (mu ** 2 / sigma).sum(-1)) # second term in girsanov
            else:
                a = (mu @ sigma_i * dWt[:,t_idx] ).sum(-1)                # first term in girsanov
                b = - 1/2 * dt * ( (mu @ sigma_i * mu).sum(-1)) # second term in girsanov

            mn = torch.distributions.multivariate_normal.MultivariateNormal(self.pmu, self.ps.abs())
            self.g = lambda x: mn.log_prob(x)
            g = self.g(Wt[:,-1,:])             # boundary condition 

            like += (g + a + b + v )

        return like.mean()

    def p(self, x):
        # x shape is batch size x d
        dt = self.t[1] - self.t[0]
        dWt = self.dWt
        Wt  = dWt.cumsum(-2) + x.unsqueeze(1) # compute brownian path up to time T
        Wt[:,:,0,:] = x
        Wt.requires_grad = True
        mu  = self.mu(Wt, Wt)
        tr_dmu = self.trace_grad(mu, Wt, N=100)  # trace of jacobian
        v = tr_dmu
        a = (mu * dWt).sum(-1).sum(-1)                # first term in girsanov
        b = - 1/2 * dt * ( (mu ** 2).sum(-1)).sum(-1) # second term in girsanov
        mn = torch.distributions.multivariate_normal.MultivariateNormal(self.pmu, self.ps)
        self.g = lambda x: mn.log_prob(x)
        g = self.g(Wt[:,:,-1,:])             # boundary condition 

        return ( g + a + b + v ).exp().mean(0)

    def trace_grad(self, mu, Wt, t, N=5):

        # Hutchinson's trace trick
        dmu = 0
        dt = self.t[1] - self.t[0]
        for _ in range(N):
            mu  = self.mu(Wt, Wt,t)
            v = torch.randn_like(mu)
            dmu += dt * (v * grad(mu, Wt, grad_outputs=v, create_graph=True)[0]).sum(-1) / N

        return dmu

    def sample(self, x0, N=500, dt=0.001):

        # basic Euler-Maruyama routine

        T = self.t[-1]
        Nt = int(T/dt)

        #x0_ = torch.randn(N, self.d).type_as(self.dWt)  #*0.01
        x0_ = self.pmu + torch.randn(N, self.d).type_as(self.dWt) @ self.ps.abs()
        x0 = x0_.clone()


        t_fine = torch.linspace(0,T,Nt).to(self.pmu.device)

        if self.sigma.shape[0] != self.d: # non diagonal noise
            self.s[self.si[0,:], self.si[1,:]] = self.sigma.abs()
            sigma = self.s

        for idx, _ in enumerate(t_fine):
            if self.sigma.shape[0] == self.d:
                x1 = x0 - self.mu(x0, x0, t_fine[idx]) * dt + self.sigma.abs().detach() * np.sqrt(dt) * torch.randn(N,self.d).to(self.dWt.device) #self.dWt[:N,0,idx,:]
            else:
                x1 = x0 - self.mu(x0, x0, t_fine[idx]) * dt + np.sqrt(dt) * torch.randn(N,self.d).to(self.dWt.device) @ sigma #self.dWt[:N,0,idx,:]
            x0 = x1

        return x1, x0_

    def configure_optimizers(self):

        opt_params = [{'params': list(self.mu.parameters())+[self.pmu, self.ps, self.sigma], 'lr': self.lr_mu}]
        #opt_params = [{'params': list(self.mu.parameters()) + [self.sigma, self.pmu], 'lr': self.lr_mu}]

        optimizer = torch.optim.AdamW(opt_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998)

        # TODO: see if these help

        return {'optimizer' : optimizer, 'scheduler': scheduler}#, 'scheduler': scheduler, 'monitor' : 'train_loss'}