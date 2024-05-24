import sympy
from sympy import sympify, lambdify
import numpy as np
import scipy as sp
import torch
import dcor

def influence_func_bump(r,center=0.5,width=1,squeeze=0.01):
    # Opinion Dynamic Model Kernel
    dim = r.shape[1]
    phi = np.zeros(r.shape)
    length = r.shape[0]
    
    bound_below = -width/2+center<r
    bound_above = r<width/2+center
    bounded = bound_below*bound_above
    r = np.clip(r*bounded, a_min=1e-12, a_max=None)
    phi = bounded * np.exp(-squeeze/(1-(2/width*(r-center))**2))
        
    return phi

def simu_drift(fcn_Ku, fcn_h, y_samps, x_obs, ts, **params):
    x_init     = params.setdefault('x_init',    [0,0])
    k          = params.setdefault('k',         [1,1])
    influence  = params.setdefault('influence', False)
    atlas      = params.setdefault('atlas', False)
    intfire    = params.setdefault('intfire', False)
    partition   = params.setdefault('partition', None)
    
    k_list = None
    if partition is not None:
        assert len(partition)+1 == len(k)
        k_list = k
        k_list = np.array(k_list)
        
    if influence: 
        center    = params.setdefault('center', 2)
        width     = params.setdefault('width', 2.5)
        squeeze   = params.setdefault('squeeze', 0.01)
        
    if intfire:
        thres       = params.setdefault('thres', [-1, 1, 0])
        eps         = params.setdefault('eps', [0.1, 0.1, 0.1])
        lambd       = params.setdefault('lambd', [1, 1, 1])
        alph        = params.setdefault('alph', [0.38, 0.38, 0.38])
        
        eps   = np.array([eps]).flatten()
        alph  = np.array([alph]).flatten()
        lambd = np.array([lambd]).flatten()
        thres = np.array([thres]).flatten()
    
    Ku_s      = sympy.sympify(fcn_Ku)
    h_s       = sympy.sympify(fcn_h)
    n_vars = x_obs.shape[-1]
    x = sympy.symbols(["x{}".format(n) for n in range(n_vars*2 + 1)])
    if n_vars == 2:
        x = sympy.symbols([x for x in ['t','X', 'Y', 'xi','yi']])
    if atlas:
        x = sympy.symbols([x for x in ['t','X','x0']])
    if intfire:
        x = sympy.symbols(["x{}".format(n) for n in range(n_vars*2 + 1)])
        
    Ku    = sympy.lambdify(x,Ku_s, "numpy") 
    h     = sympy.lambdify(x,h_s, "numpy") 
    drift = np.zeros(x_obs.shape)
    M     = np.zeros(x_obs.shape)

    if k_list is not None:
        k_index = np.searchsorted(partition, y_samps[:,0,:], side='left', sorter=None)
        k = np.array([k_list[k_index[:, n]] for n in range(n_vars)])
    else:
        k = np.ones((n_vars, y_samps.shape[0]))*k
    
    for i, t in enumerate(ts):
        y_t = np.array( y_samps[:, i, :] )
        y_in_dim = [y_t[:,dim] for dim in range(y_t.shape[-1])]
        
        x_t = np.array( x_obs[:, i, :] )
        if influence == True:
            for j in range(drift.shape[0]):
                r_list = x_t[j] - y_t
                drift[j,i,:] = -(r_list * influence_func_bump(abs(r_list), center, width, squeeze)).mean(0)
        elif atlas == True:
            x_argsort_mean = x_t[:,0].argsort().argsort()/(drift.shape[0]+1)
            drift[:,i,:] = np.array(Ku(t,x_t[:,0],x_argsort_mean)).T
        elif intfire == True:
            epsilon_ball_below = np.greater(x_obs[:,i,:], thres-eps)
            epsilon_ball_above = np.less(x_obs[:,i,:], thres+eps)
            M[:,i,:] = np.array(x_obs[:,i,:] * (epsilon_ball_above*epsilon_ball_below))
            drift[:,i,:] = (-x_t*lambd + \
                          alph * np.mean(M[:,0:i+1,:],1)/n_vars - \
                          np.sum(M[:,0:i+1,:], 1))
        else:
            for j in range(drift.shape[0]):
                drift[j,i,:] = (h(t, *y_in_dim, *x_t[j]) \
                                + np.mean(k.T*np.array(Ku(t, *y_in_dim, *x_t[j])).T, axis=0))
    
    return drift

def MLP_drift(driftMLP, y_samps, x_obs, device, ts, label_x, label_y, **params):
    drift = torch.zeros(x_obs.shape).to(device)
    for i, t in enumerate(ts):
        y_t = y_samps[:, i, :].float().to(device)
        x_t = x_obs[:, i, :].float().to(device)
        t_t = t.float().to(device)
        drift[:,i,:] = driftMLP(x_t, y_t, t_t, label_x.to(device) if label_x is not None else None, 
                                label_y.to(device) if label_y is not None else None)

    return drift

def generate_path(driftMLP,  ts, device, y_samps=None, label_x=None, 
                  label_y=None, NF=None, x_init=None, **params):
    num_samples = params.setdefault('num_samples', 100)
    n_vars      = params.setdefault('n_vars', 2)
    sigma       = params.setdefault('sigma', 1)
    N           = params.setdefault('n_points', 100)
    influence   = params.setdefault('influence', False)
    
    if NF:
        NF.to(device)
        num_NF_samples = params.setdefault('num_NF_samples', 100)
    driftMLP.to(device)
    try:
        ts = torch.from_numpy(ts).float().to(device)
    except:
        ts = ts.to(device)
    
    dt = (ts[1] - ts[0])
    xsamps = torch.zeros(num_samples, ts.shape[0], n_vars).to(device)
    for i, t in enumerate(ts):
        if i == 0:
            # different ways to init
            if influence == True:
                # for opinion dynamic model
                xsamps[:,0,:] = torch.linspace(-2,2, 100, device=device).repeat(2,1).T
            if x_init is not None and (np.array(x_init.shape) == np.array(tuple([num_samples, n_vars]))).all():
                # for given init
                # initiate with different types
                try:
                    xsamps[:,0,:] = torch.from_numpy(x_init).to(device)
                except TypeError:
                    xsamps[:,0,:] = x_init.to(device)
            elif NF is not None and x_init is None:
                # using NF init
                xsamps[:,0,:] = NF.sample(num_samples=num_samples, 
                                          cond_inputs=torch.zeros(num_samples,1).to(device))
            continue
        # Sampling
        x_t = xsamps[:,i-1,:]
        y_t = None
        t_t = t
        
        # different ways of using y_t
        if NF:
            y_t = NF.sample(num_samples=num_NF_samples,
                            cond_inputs=t.repeat(num_NF_samples,1).to(device))
        elif driftMLP.W_0_hidden == 0 and NF is None and y_samps is None:
            y_t = x_t
        elif driftMLP.W_0_hidden == 0 and NF is None and y_samps is not None:
            y_t = y_samps[:,i-1,:].float().to(device)
            
        drift = driftMLP(x_t, y_t, t_t, label_x.to(device)if label_x is not None else None, 
                         label_y.to(device) if label_y is not None else None)
        sigma = driftMLP.sigma_forward(t_t)
        sigma = torch.clip(sigma, min=1e-3)
        x_t = torch.normal(mean=x_t + drift*dt, std=sigma*torch.sqrt(dt).flatten())
        xsamps[:,i,:] = x_t
        
        #del x_t
        #del y_t
        #del t_t
        #
        #torch.cuda.empty_cache()
            
    return xsamps
 
def crps(test, gen):
    crps = 0
    if len(test.shape) > 3 and len(gen.shape) > 3:
        test = test.reshape(test.shape[0]*test.shape[1], *(test.shape[2:]))
        gen = gen.reshape(gen.shape[0]*gen.shape[1], *(gen.shape[2:]))
        
    for i in range(test.shape[1]):
        q = gen[:,i,0]
        crps += np.mean(abs(np.repeat(q, test.shape[0]) - np.tile(test[:,i,0], q.shape[0])))
        crps += -0.5*np.mean(abs(np.repeat(q, q.shape[0]) - np.tile(q, q.shape[0])))
    return crps/test.shape[1]

def evaluation(test, gen):
    metric_dict = {}
    metric_dict["MSE"] = np.square(test - gen).mean()
    metric_dict["MSE_by_axis"] = np.square(test - gen).mean(0)
    metric_dict["MAE"] = np.abs(test - gen).mean()
    metric_dict["MAE_by_axis"] = np.abs(test - gen).mean(0)
    metric_dict["percent_error"] = np.abs((test - gen)/np.clip(test, a_min=1e-6, a_max=None)).mean()
    metric_dict["percent_error_by_axis"] = np.abs((test - gen)/np.clip(test, a_min=1e-6, a_max=None)).mean(0)
    
    metric_dict["energy_distance"] = np.sqrt(dcor.energy_distance(test[:,-1], np.array(gen[:,-1])))
    metric_dict["energy_distance_by_time"] = [np.sqrt(dcor.energy_distance(test[:,i], np.array(gen[:,i]))) for i in range(gen.shape[1])]

    if gen.shape[-1] == 1:
        metric_dict["CRPS"] = crps(test, gen)
                         
    return metric_dict


def pad_data(data, window_size):
    start = 0
    end = 1
    sequences = []
    labels = []
    while end < data.shape[-2]:
        if end - start < window_size:
            pad_shape = list(data.shape)
            pad_shape[-2] = window_size
            pad_output = np.zeros(tuple(pad_shape))
            pad_output[..., start:end, :] = data[..., start:end, :]
            labels.append(data[..., end, :])
            end += 1
        else:
            pad_output = data[..., start:end, :]
            labels.append(data[..., end, :])
            end += 1
            start += 1
            
        sequences.append(pad_output)
        
    sequences = np.array(sequences)
    sequence = sequences.reshape((sequences.shape[0]*sequences.shape[1], *sequences.shape[2:]))
    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0]*labels.shape[1], *labels.shape[2:]))
    
    return torch.from_numpy(sequence), torch.from_numpy(np.array(labels))

from collections import defaultdict

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def make_particle_label(x0, partition=None):
    if partition is None:
        particle_labels = torch.tensor(range(x0.shape[0])).reshape(-1,1)
        return particle_labels
    else:
        if partition is not None:
            k_index = np.searchsorted(partition, x0, side='left', sorter=None)
            labels = dict([])
            count = 0
            particle_labels = []
            for i, item in enumerate(k_index):
                curr_key_list = list(map(str, item))
                curr_key = ''.join(curr_key_list)
                if curr_key not in labels:
                    labels[curr_key] = count
                    count += 1
                particle_labels.append(labels[curr_key])
            
        return torch.from_numpy(np.array(particle_labels))