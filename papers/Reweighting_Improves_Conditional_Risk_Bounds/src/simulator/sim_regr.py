"""
synthetic regression data generator
"""

import numpy as np
import matplotlib.pyplot as plt

class rbfNetwork():
    """
    conduct the mapping x -> f(x) where f is parametrized by a radial basis function network: R^d -> R
    in our specific case, d = 1 (since x is 1d)
    - f(x) = \sum_{i=1}^K a_i \exp{ -beta_i * |x - c_i|^2 }
    - K is the number of hidden layer neurons (hidden_dim)
    """
    def __init__(
        self,
        K = 100,                    ## number of bases
        c_range = [0,10],           ## range for the centers
        beta_range = [0.01,0.02],
        target_max = np.sqrt(10)
    ):
        self.input_dim = 1
        self.output_dim = 1
        self.K = K
        self.c_range = c_range
        self.beta_range = beta_range
        self.target_max = target_max
        
        self.assign_network_params()
        
    def assign_network_params(self):
        """assign network parameters"""
        ## centers (c_i's)
        self.cs = np.random.uniform(low=self.c_range[0],high=self.c_range[1],size=(self.K,self.input_dim))
        ## scales (beta_i's)
        self.betas = np.random.uniform(low=self.beta_range[0],high=self.beta_range[1],size=(self.K,))
        ## weights to the output layer (i.e., the a_i's), using normalized xavier initialization
        self.weights = np.random.uniform(low=-np.sqrt(6)/np.sqrt(self.K+self.output_dim),
                                 high=np.sqrt(6)/np.sqrt(self.K+self.output_dim),
                                 size=(self.output_dim,self.K))
    def convert(self,x):
        """convert a "batch" of x; x.shape = [B, self.input_dim]"""
        if x.ndim == 1:
            x = np.reshape(x,(-1,1))
        bases = np.zeros((x.shape[0],self.K))
        for k in range(self.K):
            bases[:,k] = np.exp(-self.betas[k] * np.sum((x - self.cs[k,:])**2,axis=1))
        fx = np.matmul(bases,self.weights.transpose())
        
        if self.target_max is not None:
            fx_norm_max = max(np.linalg.norm(fx,axis=-1))
            scaler = self.target_max / fx_norm_max
            fx = scaler * fx
            print(f'final result rescaled to match target_max={self.target_max:.3f}; scaler={scaler: .3f}')
        return fx.squeeze()

class SimRegr():

    def __init__(self, params, seed=None):

        self.params = params
        self.seed = seed if seed is not None else np.random.choice(np.arange(100),1)[0]
        
    def generate_dataset(self, ds_id, n, num_of_replicates=10):

        fn = getattr(SimRegr, f'_make_{ds_id}')
        description = self.params['description']

        data_with_replica, ground_truth = fn(n, self.params, num_of_replicates, seed=self.seed)
        print(f'seed={self.seed}; total_sample_size={n}; num_of_replicas={num_of_replicates}')
        print(f'ds_id={ds_id}; description={description}')
        return {'data_with_replica': data_with_replica, 'ground_truth': ground_truth, 'seed': self.seed}

    @staticmethod
    def _make_quadraticVar(n,params,num_of_replicates=10,seed=None):
        
        if seed is not None:
            np.random.seed(seed)

        boundary_extension = params.get('boundary_extension', 0)
        
        xlow = params['xlow'] - boundary_extension
        xhigh = params['xhigh'] + boundary_extension
        sigma, xi_max = params['sigma'], params.get('xi_max', None)

        ### effective sample size
        n = int(n* (xhigh - xlow)/(params['xhigh'] - params['xlow']))

        data_with_replica, ground_truth = [], []
        for i in range(num_of_replicates):

            x = np.random.uniform(xlow,xhigh,size=n)
            true_mean, true_var = x * np.sin(x), sigma**2 * (1+x**2)

            xi = sigma * np.random.normal(size=(n,2))
            if xi_max is not None:
                np.clip(xi, -xi_max, xi_max, out=xi)
            e = x * np.squeeze(xi[:,0]) + np.squeeze(xi[:,1])
            y = true_mean + e

            x, y, e = np.reshape(x,(-1,1)), np.reshape(y,(-1,1)), np.reshape(e,(-1,1))
            true_mean, true_var = np.reshape(true_mean,(-1,1)), np.reshape(true_var,(-1,1))
            assert y.shape == e.shape

            data_with_replica.append((x,y,e))
            ground_truth.append((true_mean,true_var))

        return data_with_replica, ground_truth

    @staticmethod
    def _make_rbfVar(n,params,num_of_replicates=10,seed=None):
        """
        g(x) = radial basis network on x; true_var = max(0.1, g(x) ** 2)
        """
        if seed is not None:
            np.random.seed(seed)

        boundary_extension = 0 if params.get('boundary_extension', None) == None else 0
        xlow = params['xlow'] - boundary_extension
        xhigh = params['xhigh'] + boundary_extension
        xi_max = params.get('xi_max', None)

        K, c_range, beta_range = params['K'], params['c_range'], params['beta_range']
        target_max = np.sqrt(10)

        rbf = rbfNetwork(K=K, c_range=c_range, beta_range=beta_range,target_max=target_max)

        data_with_replica, ground_truth = [], []
        for i in range(num_of_replicates):

            x = np.random.uniform(xlow,xhigh, size=n)
            true_mean = x * np.sin(x)

            true_var = np.maximum(0.1, rbf.convert(x)**2)

            xi = 1/np.sqrt(8/(8-2)) * np.random.standard_t(df=8,size=(n,1)) ## xi
            if xi_max is not None:
                np.clip(xi, -xi_max, xi_max, out=xi)
            e = np.sqrt(true_var) * np.squeeze(xi[:,0])
            y = true_mean + e
            
            x, y, e = np.reshape(x,(-1,1)), np.reshape(y,(-1,1)), np.reshape(e,(-1,1))
            true_mean, true_var = np.reshape(true_mean,(-1,1)), np.reshape(true_var,(-1,1))
            assert y.shape == e.shape

            data_with_replica.append((x,y,e))
            ground_truth.append((true_mean,true_var))

        return data_with_replica, ground_truth

    @staticmethod
    def view_dataset(data, save_file_as=""):
        
        x, y, true_mean, true_var = data['x'], data['y'], data['true_mean'], data['true_var']
        sample_size = x.shape[0]
        
        fig, axs = plt.subplots(1,2,figsize=(10,2),constrained_layout=True)
        
        axs[0].scatter(x,y,label='observed_y',s=2,color='#ADD8E6',marker='.')
        axs[0].scatter(x,true_mean,label='true_mean',s=1,color='orange',marker='.')
        axs[0].scatter(x,true_mean+2*np.sqrt(true_var),label='mean+2std',s=1,color='black',marker='.')
        axs[0].scatter(x,true_mean-2*np.sqrt(true_var),label='mean-2std',s=1,color='black',marker='.')
        axs[0].set_title(f'true mean vs. x, total sample size = {sample_size}')
        
        x, true_var = zip(*sorted(zip(x, true_var)))
        axs[1].plot(x,true_var,label='true_var',color='orange')
        axs[1].set_title(f'true variance vs. x, total sample_size = {sample_size}')
        
        if len(save_file_as):
            fig.savefig(save_file_as,facecolor='w')
        
        plt.close('all')
