#@title Defining a time-dependent score-based model (double click to expand or collapse)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import functools
import os
import argparse
datasets = ["hepmass","cortex", "power", "miniboone", "ethylene_CO"]
# Hyper parameters, setup experiment folder
parser = argparse.ArgumentParser()
parser.add_argument("--sig", help="sigma")
parser.add_argument("--lr", help="lr")
parser.add_argument("--bs", help="batch_size")
parser.add_argument("--ns", help="num_steps")
parser.add_argument("--d", default='cuda:0', help="device")

args = parser.parse_args()

device = args.d
sigma  = float(args.sig)
lr     = float(args.lr)
batch_size = int(args.bs)
num_steps  = int(args.ns)

exp_folder = "sig={}_lr={}_batch={}_num_step={}/".format(sigma, lr, batch_size, num_steps)
if not os.path.exists(exp_folder):
  os.makedirs(exp_folder)
  os.makedirs(exp_folder + "/plot")
  os.makedirs(exp_folder + "/model")
  

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps.""" 
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, in_dim, out_dim, n_layers = 8, embed_dim=128):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    
    # Todo: Change to MLP
    
    # The swish activation function
    self.act = nn.SiLU()
    self.Linear1 = nn.Linear(in_dim, embed_dim)
    MLP_layers = []
    for i in range(n_layers-2): 
        MLP_layers.append(nn.Linear(embed_dim, embed_dim))
        MLP_layers.append(self.act)
        
    MLP_layers.append(nn.Linear(embed_dim, out_dim))
    self.marginal_prob_std = marginal_prob_std
    self.MLP = nn.Sequential(*MLP_layers)
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    h1 = self.act(self.Linear1(x)) + embed  
    ## Incorporate information from t
    h = self.MLP(h1)

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None]
    return h


#@title Set up the SDE

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

def marginal_prob_std(t, sigma=1):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma=1):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)
  

marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

#@title Define the loss function (double click to expand or collapse)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
  return loss


#@title Training (double click to expand or collapse)


  
  
#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff,
                           in_dim,  
                           batch_size=64, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, in_dim, device=device) \
    * marginal_prob_std(t)[:, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in time_steps:      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x

#@title Define the Predictor-Corrector sampler (double click to expand or collapse)

signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
#@param {'type':'integer'}
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=num_steps, 
               snr=signal_to_noise_ratio,                
               device='cuda',
               eps=1e-3):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in time_steps:      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean


#@title Define the ODE sampler (double click to expand or collapse)

from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x















###################### Training Loop ######################


#@title Sampling (double click to expand or collapse)
import dcor

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import matplotlib.pyplot as plt
import pickle


mean_list = []
sd_list = []
for dataset in datasets:
    ed_list = []
    for run in range(10):
        n_epochs =   500#@param {'type':'integer'}
        ## size of a mini-batch
        batch_size =  10 #@param {'type':'integer'}

        N_train = 100; N_val = 500; N_test = 500
        tot_N = N_train + N_val + N_test
        if 'cortex' == dataset:
            data = pd.read_csv("/hpc/group/tarokhlab/hy190/MV-SDE/data/realGen/Data_Cortex_Nuclear.csv").values[:,2:78].astype(float)
            data = np.nan_to_num(data)
        if 'miniboone' == dataset:
            data = np.load("/hpc/group/tarokhlab/hy190/MV-SDE/data/realGen/miniboone.npy")
        if 'ethylene_CO' == dataset:
            data = pd.read_pickle("/hpc/group/tarokhlab/hy190/MV-SDE/data/realGen/ethylene_CO.pickle")
            data.drop("Meth", axis=1, inplace=True)
            data.drop("Eth", axis=1, inplace=True)
            data.drop("Time", axis=1, inplace=True)
            C = data.corr()
            A = C > 0.98
            B = A.values.sum(axis=1)
            while np.any(B > 1):
                col_to_remove = np.where(B > 1)[0][0]
                col_name = data.columns[col_to_remove]
                data.drop(col_name, axis=1, inplace=True)
                C = data.corr()
                A = C > 0.98
                B = A.values.sum(axis=1)
            data = (data - data.mean()) / data.std()
            data = data.values.astype(float)
        if 'power' == dataset:
            data = np.load("/hpc/group/tarokhlab/hy190/MV-SDE/data/realGen/power_data.npy")
            N = data.shape[0]
            data = np.delete(data, 3, axis=1)
            data = np.delete(data, 1, axis=1)
            voltage_noise = 0.01 * np.random.rand(N, 1)
            gap_noise = 0.001 * np.random.rand(N, 1)
            sm_noise = np.random.rand(N, 3)
            time_noise = np.zeros((N, 1))
            noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
            data = data + noise
            data = data.astype(float)
        if 'hepmass' == dataset:
            data = pd.read_csv(filepath_or_buffer="/hpc/group/tarokhlab/hy190/MV-SDE/data/realGen/hepmass_1000_train.csv",
                        index_col=False)
            data = data[data[data.columns[0]] == 1]
            data = data.drop(data.columns[0], axis=1)
            data = data.values.astype(float)
        # subsampling

        index = np.random.choice(list(range(data.shape[0])), np.min([tot_N, data.shape[0]]), False)
        data = data[index]
        
        data_train = data[0:N_train]
        data_validate = data[N_train:N_train + N_val]
        if "hepmass" == dataset:
            data_test = pd.read_csv(filepath_or_buffer="/hpc/group/tarokhlab/hy190/MV-SDE/data/realGen/hepmass_1000_test.csv",
                                        index_col=False)
            data_test = data_test[data_test[data_test.columns[0]] == 1]
            data_test = data_test.drop(data_test.columns[0], axis=1)
            # Because the data set is messed up!
            data_test = data_test.drop(data_test.columns[-1], axis=1)
            data_test = data_test.values.astype(float)
            np.random.shuffle(data_test)
            data_test = data_test[:N_test]
        else:
            data_test = data[N_train + N_val:data.shape[0]]
        # normalize
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s
            
        train_tensor = torch.from_numpy(data_train)
        data_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        test_tensor = torch.from_numpy(data_test)
        test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=True)
        
        in_dim = out_dim = data_train.shape[-1]
        score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, in_dim=in_dim, out_dim=out_dim)
        score_model = score_model.to(device)
        driftMLP_param=0
        driftMLP_param += sum(p.numel() for p in score_model.parameters())
        print("Total Number of Parameters: {}".format(driftMLP_param))
        optimizer = Adam(score_model.parameters(), lr=lr)
        tqdm_epoch = tqdm.tqdm(range(n_epochs))
        for epoch in range(n_epochs):
            avg_loss = 0.
            num_items = 0
            for x in data_loader:
                x = x.to(device).float()
                loss = loss_fn(score_model, x, marginal_prob_std_fn)
                optimizer.zero_grad()
                loss.backward()    
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            tqdm_epoch.update(1)
            # Update the checkpoint after each epoch of training.
            torch.save(score_model.state_dict(), exp_folder + 'model/{}_{}_ckpt.pth'.format(dataset, run))
        tqdm_epoch.close()

        ## Load the pre-trained checkpoint from disk.
        device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
        ckpt = torch.load(exp_folder + 'model/{}_{}_ckpt.pth'.format(dataset, run), map_location=device)
        score_model.load_state_dict(ckpt)

        sample_batch_size = data_test.shape[0] #@param {'type':'integer'}
        sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

        ## Generate samples using the specified sampler.
        samples = sampler(score_model, 
                        marginal_prob_std_fn,
                        diffusion_coeff_fn, 
                        batch_size=sample_batch_size, 
                        in_dim=in_dim,
                        device=device).detach().cpu().numpy()
        ed = np.sqrt(dcor.energy_distance(data_test, samples))
        ed_list.append(ed)
        print(ed)
        
        plt.figure(figsize=(5,5))
        plt.scatter(data_test[:100, 0], data_test[:100, 1])
        plt.scatter(samples[:100, 0], samples[:100, 1])
        plt.legend(["True", "Generated"])
        plt.savefig(exp_folder + "plot/{}_{}".format(dataset, run))
        
    print(dataset, np.array(ed_list).mean(), np.array(ed_list).std())
    
    mean_list.append(np.array(ed_list).mean())
    sd_list.append(np.array(ed_list).std())
    
with open(exp_folder + 'saved_stats.pkl', 'wb') as f:
    pickle.dump((mean_list, sd_list), f)
    f.close()
print("\n\n\n")
print("--------------------------")
print(datasets)
print(mean_list)
print(sd_list)