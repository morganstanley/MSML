import numpy as np
import sympy
from sympy import sympify, lambdify
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn import datasets
import ot

from utils import influence_func_bump

def sim_process_mckean(fcn_Ku,fcn_h,fcn_sigma,**params):
    """
    Sampling Stochastic Mckean Process
    Currently Hardcoded for 2D
    It can be optimizerd with np.repeat and np.tile
    """
    
    seed        = params.setdefault('seed')
    np.random.seed(seed=seed)

    n_particles = params.setdefault('n_particles', 50)
    N           = params.setdefault('n_points', 100)
    t_init      = params.setdefault('t0', 0)
    t_end       = params.setdefault('tn', 5)
    x_init      = params.setdefault('x_init', np.array([0,0]))
    init_var    = params.setdefault('init_var', 1)
    init_mu     = params.setdefault('init_mu', [0,0])
    k           = params.setdefault('k', [1,1])
    n_vars      = params.setdefault('n_vars', 2)
    influence   = params.setdefault('influence', False)
    grid_init   = params.setdefault('grid_init', False)
    grid_space  = params.setdefault('grid_space', 6)
    irregular   = params.setdefault('irregular', False)
    partition   = params.setdefault('partition', None)
    n_jumps     = params.setdefault('n_jumps', 0)
    
    def dW(delta_t,n_vars,flight=False): 
        """Sample a random number at each call."""
        #if t_dist:
        #    return np.sqrt(delta_t) * np.random.standard_t(df=4, size=(n_vars,1))
        if flight:
            return np.random.levy(loc = np.zeros((n_vars,1)), scale = np.sqrt(delta_t))
        return np.random.normal(loc = np.zeros((n_vars,1)), scale = np.sqrt(delta_t))
    
    n_vars = len(fcn_Ku.split(','))
    
    k_list = None
    if partition is not None:
        assert len(partition)+1 == len(k)
        k_list = k
        k_list = np.array(k_list)
    
    if influence: 
        center    = params.setdefault('center', 2)
        width     = params.setdefault('width', 2.5)
        squeeze   = params.setdefault('squeeze', 0.01)
        grid_init = True
        
    if irregular:
        n_irreg   = params.setdefault('n_irreg', 1)
        assert n_irreg < N
    
    Ku_s      = sympy.sympify(fcn_Ku)
    h_s      = sympy.sympify(fcn_h)
    sigma_s   = sympy.sympify(fcn_sigma)
    
    
    x = sympy.symbols(["x{}".format(n) for n in range(n_vars*2 + 1)])
    if n_vars == 2:
        x = sympy.symbols([x for x in ['t','X', 'Y', 'xi','yi']])
        
    Ku    = sympy.lambdify(x,Ku_s, "numpy")
    h     = sympy.lambdify(x,h_s, "numpy") 
    sigma = sympy.lambdify(x,sigma_s, "numpy")

    dt = (t_end - t_init) / N
    ts = np.arange(t_init, t_end, dt)
    t_jump = np.random.choice(ts, int(n_jumps), False).tolist()
    xs = np.zeros( (n_particles, N, n_vars) )
    irreg_t = []
    # Different initializations
    if grid_init == True:
        xs[:,0,:] = np.repeat(np.linspace(-grid_space,grid_space,n_particles), n_vars).reshape(n_particles, n_vars)
    elif (np.array(np.array(x_init).shape) == np.array(tuple([n_particles, n_vars]))).all():
        xs[:,0,:] = x_init
    else: 
        xs[:,0,:] = np.random.normal(init_mu,init_var,(n_particles, n_vars))


    if k_list is not None:
        k_index = np.searchsorted(partition, xs[:,0,:], side='left', sorter=None)
        k = np.array([k_list[k_index[:, n]] for n in range(n_vars)])
    else:
        k = np.ones((n_vars, n_particles))*k
    for i, t in enumerate(ts):
        if i == 0:
            continue
        
        x = np.array( xs[:, i-1, :] )
        x_in_dim = [x[:,dim] for dim in range(x.shape[-1])]
        if influence == True:
            for j in range(n_particles):
                r_list = x[j] - x
                xs[j,i,:] =  x[j] - \
                             np.mean(r_list * 1 *influence_func_bump(abs(r_list), center, width, squeeze), axis=0) * dt + \
                             np.squeeze(np.array(sigma(t,*x_in_dim, *x[j])) * dW(dt, n_vars)).reshape(1,n_vars)
                
                
        elif influence == False:
            for j in range(n_particles):
                xs[j,i,:] = x[j] + \
                (h(t, *x_in_dim, *x[j]) + \
                np.mean(k.T*np.array(Ku(t,*x_in_dim, *x[j])).T, axis=0)) * dt + \
                np.squeeze(np.array(sigma(t, *x_in_dim, *x[j])) * dW(dt, n_vars)).reshape(1,n_vars)
                if n_jumps > 0 and t in set(t_jump):
                    xs[j,i,:] += np.exp(np.random.uniform(2,3, size=2))
    if not irregular:
        irreg_t = None
    else:
        #irreg_t.append(N-1)
        ts_mask = np.cumsum(np.random.exponential(scale=t_end/n_irreg, size=n_irreg))
        unique_steps = list(set(abs(ts_mask.reshape(n_irreg,1) - np.tile(ts,(n_irreg,1))).argmin(1)))
        unique_steps.append(N-1)
        irreg_t = list(set(unique_steps))
        irreg_t.sort()
        if 0 == irreg_t[0]:
            irreg_t = irreg_t[1:]
        
    return xs, ts, n_particles, irreg_t



def sim_process_meanfield_atlas(fcn_Ku,fcn_sigma,**params):
    """
    Sampling Stochastic Meanfield Atalas Process
    Currently Hardcoded for 1D
    It can be optimizerd with np.repeat and np.tile
    """
    def dW_batch(delta_t,n_vars,n_particles, flight=False): 
        """Sample a random number at each call."""
        if flight:
            return np.random.levy(loc = np.zeros((n_particles, n_vars)), scale = np.sqrt(delta_t))
        return np.random.normal(loc = np.zeros((n_particles, n_vars)), scale = np.sqrt(delta_t))
    seed        = params.setdefault('seed')
    np.random.seed(seed=seed)
    n_particles = params.setdefault('n_particles', 50)
    N           = params.setdefault('n_points', 100)
    t_init      = params.setdefault('t0', 0)
    t_end       = params.setdefault('tn', 5)
    x_init      = params.setdefault('x_init', np.array([0,0]))
    init_var    = params.setdefault('init_var', 1)
    k           = params.setdefault('k', [1,1])
    grid_init   = params.setdefault('grid_init', False)
    grid_space  = params.setdefault('grid_space', 6)
    irregular   = params.setdefault('irregular', False)
        
    if irregular:
        n_irreg   = params.setdefault('n_irreg', 1)
        assert n_irreg < N
    
    Ku_s      = sympy.sympify(fcn_Ku)
    sigma_s   = sympy.sympify(fcn_sigma)
    
    x = sympy.symbols([x for x in ['t','X','x0']])
        
    Ku    = sympy.lambdify(x,Ku_s, "numpy")
    sigma = sympy.lambdify(x,sigma_s, "numpy")

    dt = (t_end - t_init) / N
    ts= np.arange(t_init, t_end, dt)
    xs = np.zeros( (n_particles, N, 1) )
    irreg_t = []
    for i, t in enumerate(ts):
        if i == 0:
            # Different initializations
            if grid_init == True:
                xs[:,0,:] = np.linspace(-grid_space,grid_space,n_particles).reshape(n_particles,1)
            elif (np.array(np.array(x_init).shape) == np.array(tuple([n_particles, 1]))).all():
                xs[:,0,:] = x_init
            else: 
                xs[:,0,:] = np.random.normal(0,init_var,(n_particles, 1))
            continue
            
        t_step = (t-1) * dt
        x = np.array( xs[:, i-1, :] )
        x_argsort_mean = x[:,0].argsort().argsort()/(n_particles+1)
        xs[:,i,:] = x + \
                np.array(Ku(t,x[:,0],x_argsort_mean)).T * dt + \
                np.array(np.array(sigma(t,x[:,0],x_argsort_mean)).T * dW_batch(dt,1,n_particles)).reshape(n_particles,1)
            
    if not irregular:
        irreg_t = None
    else:
        ts_mask = np.cumsum(np.random.exponential(scale=t_end/n_irreg, size=n_irreg))
        unique_steps = list(set(abs(ts_mask.reshape(n_irreg,1) - np.tile(ts,(n_irreg,1))).argmin(1)))
        unique_steps.append(N-1)
        irreg_t = list(set(unique_steps))
        irreg_t.sort()
        if 0 == irreg_t[0]:
            irreg_t = irreg_t[1:]
        
    return xs, ts, n_particles, irreg_t


def sim_process_meanfield_intfire(fcn_sigma,**params):
    """
    Sampling Stochastic Meanfield Atalas Process
    Currently Hardcoded for 2D
    It can be optimizerd with np.repeat and np.tile
    """
    def dW_batch(delta_t,n_vars,n_particles, flight=False): 
        """Sample a random number at each call."""
        if flight:
            return np.random.levy(loc = np.zeros((n_particles, n_vars)), scale = np.sqrt(delta_t))
        return np.random.normal(loc = np.zeros((n_particles, n_vars)), scale = np.sqrt(delta_t))
    seed        = params.setdefault('seed')
    np.random.seed(seed=seed)
    n_particles = params.setdefault('n_particles', 50)
    N           = params.setdefault('n_points', 100)
    n_vars      = params.setdefault('n_vars', 3)
    t_init      = params.setdefault('t0', 0)
    t_end       = params.setdefault('tn', 5)
    x_init      = params.setdefault('x_init', np.array([0,0]))
    init_var    = params.setdefault('init_var', 1)
    k           = params.setdefault('k', [1,1])
    grid_init   = params.setdefault('grid_init', False)
    grid_space  = params.setdefault('grid_space', 6)
    irregular   = params.setdefault('irregular', False)
    
    thres       = params.setdefault('thres', [-1, 1, 0])
    eps         = params.setdefault('eps', [0.1, 0.1, 0.1])
    lambd       = params.setdefault('lambd', [1, 1, 1])
    alph        = params.setdefault('alph', [0.38, 0.38, 0.38])
    
    eps   = np.array([eps]).flatten()
    alph  = np.array([alph]).flatten()
    lambd = np.array([lambd]).flatten()
    thres = np.array([thres]).flatten()
    
    if irregular:
        n_irreg   = params.setdefault('n_irreg', 1)
        assert n_irreg < N
    
    x = sympy.symbols(["x{}".format(n) for n in range(n_vars*2 + 1)])
    sigma_s   = sympy.sympify(fcn_sigma)
    sigma = sympy.lambdify(x,sigma_s, "numpy")

    dt = (t_end - t_init) / N
    ts = np.arange(t_init, t_end, dt)
    xs = np.zeros( (n_particles, N, n_vars) )
    M  = np.zeros( (n_particles, N, n_vars) )
    
    irreg_t = []
    for i, t in enumerate(ts):
        if i == 0:
            # Different initializations
            if grid_init == True:
                xs[:,0,:] = np.linspace(-grid_space,grid_space,n_particles).reshape(n_particles,1)
            elif (np.array(np.array(x_init).shape) == np.array(tuple([n_particles, n_vars]))).all():
                xs[:,0,:] = x_init
            else: 
                xs[:,0,:] = np.random.normal(0,init_var,(n_particles, n_vars))
            epsilon_ball_below = np.greater(xs[:,i,:], thres-eps)
            epsilon_ball_above = np.less(xs[:,i,:], thres+eps)
            M[:,i,:] = np.array(xs[:,i,:] * (epsilon_ball_above*epsilon_ball_below))
            continue
        
        t_step = (t-1) * dt
        
        x = np.array( xs[:, i-1, :] )
        x_in_dim = [x[:,dim] for dim in range(x.shape[-1])]
        
        xs[:,i,:] = x + (-x*lambd + \
                          alph * np.mean(M[:,0:i,:],1)/n_vars - \
                          np.sum(M[:,0:i,:], 1)) * dt + \
                np.array(np.array(sigma(t, *x_in_dim, *x_in_dim)).T * \
                         dW_batch(dt,n_vars,n_particles)).reshape(n_particles,n_vars)
                         
        epsilon_ball_below = np.greater(xs[:,i,:], thres-eps)
        epsilon_ball_above = np.less(xs[:,i,:], thres+eps)
        M[:,i,:] = np.array(xs[:,i,:] * (epsilon_ball_above*epsilon_ball_below))
            
    if not irregular:
        irreg_t = None
    else:
        ts_mask = np.cumsum(np.random.exponential(scale=t_end/n_irreg, size=n_irreg))
        unique_steps = list(set(abs(ts_mask.reshape(n_irreg,1) - np.tile(ts,(n_irreg,1))).argmin(1)))
        unique_steps.append(N-1)
        irreg_t = list(set(unique_steps))
        irreg_t.sort()
        if 0 == irreg_t[0]:
            irreg_t = irreg_t[1:]
    return xs, ts, n_particles, irreg_t

def brownian_bridge(X0, X1, t0, t1, n=100, m=5, sigma=1):
    """
    Sample Brownian Bridges from arbitrary t0,  t1
    X1, X0 are both vector inputs with dimension k*1
    """
    assert X1.shape[0] == X0.shape[0]
    assert t0 < t1
    if isinstance(X1, np.ndarray):
        X1 = torch.from_numpy(X1)
    if isinstance(X0, np.ndarray):
        X0 = torch.from_numpy(X0)
        
    X1 = X1.reshape(X1.shape[0],1); X0 = X0.reshape(X0.shape[0],1)
    X1 = X1.repeat_interleave(m,dim=0); X0 = X0.repeat_interleave(m,dim=0)

    t = torch.linspace(t0,t1,n).repeat(X1.shape[0],1)
    dt = t[0,1] - t[0,0]
    dW = torch.randn_like(t) * dt.sqrt() * sigma
    W = dW.cumsum(1)
    W[:,0] = 0
    W = W + X0
    BB = W - ((t-t0)/(t1-t0) * (W[:,-1] - X1.squeeze(1)).reshape(X1.shape[0],1))
    return BB, t[0]

def brownian_bridge_nd(X0, X1, t0, t1, n=100, m=5, sigma=1):
    """
    Sample Brownian Bridges from arbitrary t0,  t1
    X1, X0 are both vector inputs with dimension K x d
    """
    assert X1.shape == X0.shape
    assert t0 < t1
    if isinstance(X1, np.ndarray):
        X1 = torch.from_numpy(X1)
    if isinstance(X0, np.ndarray):
        X0 = torch.from_numpy(X0)

    d = X1.shape[-1]
    X1 = X1.repeat_interleave(m,dim=0)
    X0 = X0.repeat_interleave(m,dim=0)

    t = torch.linspace(t0,t1,n)
    dt = t[1] - t[0]
    try:
        dW = torch.randn(X0.shape[0], n, d) * dt.sqrt() * sigma(t).clip(min=1e-3).cpu()
    except Exception as e:
        print(e)
        dW = torch.randn(X0.shape[0], n, d) * dt.sqrt() * sigma
        
    W = dW.cumsum(1)
    W[:,0] = 0
    W = W + X0.unsqueeze(1)
    BB = W - ((t.unsqueeze(0).unsqueeze(-1)-t0)/(t1-t0) * (W[:,-1] - X1).unsqueeze(1))
    return BB, t


def BB_to_data(datatype="moons", grids=False, **params):
    """
    Sample a set of brownian bridge taking standard normal to some shaped distribution
    """
    seed           = params.setdefault('seed', 0)
    min_dist       = params.setdefault('min_dist', True)
    n_samples      = params.setdefault('n_samples', 100)
    n_bridge       = params.setdefault('n_bridge', 1)
    N              = params.setdefault('n_points', 100)
    T              = params.setdefault('T', 0.1)
    mu             = params.setdefault('mu', 0)
    SD             = params.setdefault('SD', 1)
    non_stoch_eval = params.setdefault('non_stoch_eval', False)
    input_data     = params.setdefault("input_data", None)
    d              = params.setdefault("d", 100)
        
    if min_dist:
        np.random.seed(seed=seed)
    else:
        np.random.seed(None)
    if grids == False:
        data = make_data_T(datatype=datatype, n_samples=n_samples, input_data=input_data, d=d)
        norm_samp = np.random.normal(mu,SD,data.shape)
        if datatype == 'eightgauss':
            norm_samp *= 0.1
        
    if min_dist:
        try:
            data = data.detach().cpu().numpy()
        except:
            pass
        M = ot.dist(norm_samp, data)
        a, b = np.ones((data.shape[0],)) / data.shape[0], np.ones((data.shape[0],)) / data.shape[0]
        match = ot.emd(a, b, M)
        data = data[match.argmax(1)]
    else: pass
    # non non_stoch_eval OT map for evaluation
    if non_stoch_eval and min_dist:
        m = torch.from_numpy((data-norm_samp)/T)
        b = torch.from_numpy((0*data - T*norm_samp)/(0-T))
        t = torch.linspace(0,T,N)
        BB = t.unsqueeze(0).unsqueeze(-1)*m.unsqueeze(1) + b.unsqueeze(1)
        BBtj = t
    else:
        BB, BBtj  = brownian_bridge_nd(norm_samp, data, 0, T, n=N, m=n_bridge)
        BB = BB[torch.randperm(BB.shape[0])]

    return BB.detach().cpu().numpy(), BBtj.detach().cpu().numpy(), None, None

def make_data_T(datatype, n_samples, input_data=None, d=100):    
    if datatype == "circles":
        data, _ = datasets.make_circles(n_samples = n_samples, noise=0.05, factor=0.5)

    elif datatype == "moons":
        data, _ = datasets.make_moons(n_samples = n_samples, noise=0.05)
            
    elif datatype == "s":
        data, _ = datasets.make_s_curve(n_samples = n_samples, noise=0.05)
        data = data[:,[0,2]]
            
    elif datatype == "3d-s":
        data, _ = datasets.make_s_curve(n_samples = n_samples, noise=0.05)
        data = data
            
    elif datatype == 'eightgauss':
    # high dimension gaussian
        eightgauss_mu = [[2,0], [0,2], [-2,0], [0,-2], 
                        [np.sqrt(2), np.sqrt(2)], 
                        [np.sqrt(2), -np.sqrt(2)], 
                        [-np.sqrt(2), -np.sqrt(2)],
                        [-np.sqrt(2), np.sqrt(2)]]
        eightgauss = []
        # d parameter only applicable for eightgauss experiments
        for i in range(8):
            a = torch.randn(size = (n_samples,d))*0.1 + torch.Tensor(eightgauss_mu[i]*int(d/2)) # mu is 2-d, *50 to make it 100d
            eightgauss.append(a)
        data = torch.cat(eightgauss, 0).detach().cpu().numpy()
        
    elif datatype == "swissroll":
        data = datasets.make_swiss_roll(n_samples=n_samples, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
    
    elif datatype == "pinwheel":
        rng = np.random.RandomState()
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
    
    elif datatype == "2spirals":
        n = np.sqrt(np.random.rand(n_samples // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_samples // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(n_samples // 2, 1) * 0.5
        data = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        data += np.random.randn(*data.shape) * 0.1
    
    elif datatype == "checkerboard":
        x1 = np.random.rand(n_samples) * 4 - 2
        x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif datatype == "5squares":
        idx = np.random.randint(0, 5, n_samples)
        idx_zo = 1 - idx // 4
        x1 = (np.random.rand(n_samples) - 1/2) + idx_zo * (np.random.randint(0, 2, n_samples) * 4 - 2) 
        x2 = (np.random.rand(n_samples) - 1/2) + idx_zo * (np.random.randint(0, 2, n_samples) * 4 - 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1)
        
    elif datatype in set(["USPS","MNIST"]):
        if isinstance(input_data, torch.Tensor) and len(input_data.shape) > 2:
            input_data = torch.flatten(input_data, start_dim=1)
        elif isinstance(input_data, np.ndarray) and len(input_data.shape) > 2:
            input_data = torch.flatten(torch.from_numpy(input_data), start_dim=1)
        data = input_data
        
    elif input_data is not None:
        data = input_data

    return data


def BB_get_drift(datatype, grids, T=0.1, input_data=None, d=100):
    n_samples = grids.shape[0]
    data = make_data_T(datatype=datatype, n_samples=n_samples, input_data=input_data, d=d)

    d = cdist(grids, data, metric="euclidean")
    assignment_grids, assignment_data = linear_sum_assignment(d)
    
    grids = grids[assignment_grids]
    data  = data[assignment_data]

    m = torch.from_numpy((data-grids)/T)
    return m.detach().cpu().numpy()






