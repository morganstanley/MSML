"""
synthetic data generator for binary classification
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

class SimClsf():

    def __init__(self, params, seed=None):

        self.params = params
        self.seed = seed if seed is not None else np.random.choice(np.arange(100),1)[0]
        
    def generate_dataset(self, dgp_str, n, num_of_replicates=10, reshape=False):

        fn = getattr(SimClsf, f'_make_{dgp_str}')
        description = self.params['description']

        data_with_replica, ground_truth = fn(n, self.params, num_of_replicates, seed=self.seed)
        print(f'seed={self.seed}; total_sample_size={n}; num_of_replicas={num_of_replicates}')
        print(f'dgp_str={dgp_str}; description={description}')
        return {'data_with_replica': data_with_replica, 'ground_truth': ground_truth, 'seed': self.seed}

    @staticmethod
    def _make_mixtureGaussian(n, params, num_of_replicates=10, seed=None):
        
        if seed is not None:
            np.random.seed(seed)

        Sigma = np.array(params['Sigma'])

        mu_0x, mu_0 = np.array(params['mu_0x']), np.array(params['mu_0'])
        mu_1, mu_1x = np.array(params['mu_1']), np.array(params['mu_1x'])
        
        p_0x, p_0, p_1, p_1x = params['p_0x'], params['p_0'], params['p_1'], params['p_1x']
        assert np.abs(p_0x + p_0 + p_1 + p_1x - 1) < 1e-6
        
        data_with_replica, ground_truth = [], []
        for i in range(num_of_replicates):
            
            x_0x = np.random.multivariate_normal(mu_0x, Sigma, n)
            x_0 = np.random.multivariate_normal(mu_0, Sigma, n)

            x_1 = np.random.multivariate_normal(mu_1, Sigma, n)
            x_1x = np.random.multivariate_normal(mu_1x, Sigma, n)

            x = np.stack([x_0x, x_0, x_1, x_1x], axis=-1)

            membership = np.random.choice([0,1,2,3],size=n, p=[p_0x, p_0, p_1, p_1x])        ## which cluster
            mixture_x = np.array([x[idx,:,membership[idx]] for idx in range(n)])

            dist_0, dist_0x = multivariate_normal(mean=mu_0, cov=Sigma), multivariate_normal(mean=mu_0x, cov=Sigma)
            dist_1, dist_1x = multivariate_normal(mean=mu_1, cov=Sigma), multivariate_normal(mean=mu_1x, cov=Sigma)
            
            gamma = np.array([( p_1 * dist_1.pdf(x) + p_1x * dist_1x.pdf(x))/(p_1 * dist_1.pdf(x) + p_0 * dist_0.pdf(x) + p_1x * dist_1x.pdf(x) + p_0x * dist_0x.pdf(x)) for x in mixture_x])
            y_initial = 1*(gamma >= 0.5)
            
            ## start flipping
            eta, y = 1.0*(gamma >= 0.5), y_initial.copy()
            for idx, y_val in enumerate(y_initial):
                if y_val == 0 and membership[idx] == 0:
                    eta[idx] = params['flipping_prob']
                    if np.random.binomial(1,params['flipping_prob'],1) == 1:
                        y[idx] = 1
                    
            x, y = np.reshape(mixture_x,(-1,2)), np.reshape(y,(-1,1))
            gamma, eta = np.reshape(gamma,(-1,1)), np.reshape(eta,(-1,1))
            
            data_with_replica.append((x,y))
            ground_truth.append((eta,gamma,membership))

        return data_with_replica, ground_truth

    @staticmethod
    def view_dataset(data, save_file_as=""):
        
        fig, axs = plt.subplots(2,2,figsize=(10,5),constrained_layout=True)
        color_1, color_0 = '#1f77b4', '#ff7f0e'
        x = data['x']
        
        color_x = [color_1 if mem == 1 else color_0 for mem in 1*(data['eta']>=0.5)]
        axs[0,0].scatter(x[:,0],x[:,1],s=5,color=color_x,marker='.')
        axs[0,0].set_title(f'1(eta>=0.5)')
        
        color_x = [color_1 if mem == 1 else color_0 for mem in 1*(data['gamma']>=0.5)]
        axs[0,1].scatter(x[:,0],data['gamma'],s=5,color=color_x,marker='.')
        axs[0,1].set_title(f'gamma vs. x')
        
        color_x = [color_1 if mem == 1 else color_0 for mem in 1*(data['eta']>=0.5)]
        axs[1,0].scatter(x[:,0], np.abs(data['eta']-0.5),s=5, color=color_x,marker='.')
        axs[1,0].set_title(f'margin vs. x[0]')
        
        color_x = [color_1 if mem == 1 else color_0 for mem in data['y']]
        axs[1,1].scatter(x[:,0],x[:,1],s=5,color=color_x,marker='.')
        axs[1,1].set_title(f'y')
        
        if len(save_file_as):
            fig.savefig(save_file_as,facecolor='w')
        
        plt.close('all')
