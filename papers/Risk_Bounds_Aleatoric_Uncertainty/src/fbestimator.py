'''
provide functional basis then estimate
'''
import scipy
import patsy
import numpy as np
import cvxpy as cp

class polyBasis():
    """ here we consider using polynomial-basis of the following form
    f(x) = b0 + b1 * |x - c| + b2 * |x-c|^2 + ... + bk * |x-c|^k
    - taking the absolute is optional
    """
    def __init__(self, params):

        self.n_degree = params['n_degree']
        self.num_bases = self.n_degree + 1
        self.centroid = params.get('centroid',0)
        self.abs = params.get('abs', False)
        self.f_min = params.get('f_min', None)
        self.f_max = params.get('f_max', None)
    
    def get_bases(self, x):
        """x: univariate sequence of size n"""
        bases = {}
        for i in range(self.n_degree+1):
            base_val = (x - self.centroid)**i
            if self.abs:
                base_val = np.abs(base_val)
            bases[i] = base_val
        return bases
        
    def functional(self,x,args):
        bases = self.get_bases(x)
        f_val = 0
        for i, base_key in enumerate(bases.keys()):
            f_val += args[i] * bases[base_key]

        if self.f_min or self.f_max:
            f_val = np.clip(f_val, self.f_min, self.f_max)
        return f_val
        
class bSplineBasis():

    def __init__(self, params):

        self.n_degree = params['n_degree']
        self.x_min = params.get('x_min', -0.25)
        self.x_max = params.get('x_max', 10.25)
        self.num_knots = params.get('num_knots',10)
        self.include_intercept = params.get('include_intercept',True)
        self.num_bases = int((self.n_degree + 1) * self.num_knots + np.sum(list(range(self.n_degree)))) + 1 * self.include_intercept

        self.f_min = params.get('f_min', None)
        self.f_max = params.get('f_max', None)
        
        self._get_knots()

    def _get_knots(self, random = False):
        if random:
            knots = np.random.uniform(self.x_min, self.x_max, size=(self.num_knots,))
            knots.sort()
        else:
            knots = np.linspace(self.x_min, self.x_max, self.num_knots)
        self.knots = knots

    def get_bases(self, x):
        """x: univariate sequence of size n"""
        bases = {}
        for i in range(self.n_degree+1):
            bases[i] = patsy.bs(x, knots=self.knots, degree=i, include_intercept=True) ## size = (n, num_knots + i)
        if self.include_intercept:
            bases['intercept'] = np.ones((len(x),1))
        return bases

    def functional(self,x,args):

        bases = self.get_bases(x)
        bases_stacked = np.column_stack(list(bases.values()))
            
        f_val = 0
        for i in range(bases_stacked.shape[1]):
            f_val += args[i] * bases_stacked[:,i]
        
        if self.f_min or self.f_max:
            f_val = np.clip(f_val, self.f_min, self.f_max)
        return f_val

class fbEstimator():

    def __init__(self, params):

        self.supported_losstypes = ['MSE','NLL','invNLL']
        self.supported_methods = ['BFGS','L-BFGS-B','Nelder-Mead','Powell','SLSQP','COBYLA']
        assert params['method'] in self.supported_methods
        assert params['loss_type'] in self.supported_losstypes
        
        self.optimizer_engine = params.get('optimizer_engine','cvxpy')
        self.method = params['method']
        self.loss_type = params['loss_type']
        self.basis_type = params['basis_type']
        self.f_min = params.get('f_min', None)
        self.f_max = params.get('f_max', None)
        self.offset = 1e-6
        
        self._params = params
        self._create_functional()

    def _create_functional(self):

        if self.basis_type == 'poly':
            basis_class = polyBasis(self._params)
        elif self.basis_type == 'b-spline':
            basis_class = bSplineBasis(self._params)
        else:
            raise ValueError('unsupported basis_type; choose among [poly, b-spline]')

        self.functional = basis_class.functional
        self.get_bases = basis_class.get_bases
        self.num_bases = basis_class.num_bases

    def _obj(self,x,y,args):
        """ helper function for fitting method I """
        preds = self.functional(x,args)
        if self.loss_type == 'MSE':
            return np.mean((y - preds)**2)
        elif self.loss_type == 'NLL':
            return np.mean(y/(preds+self.offset) + np.log(preds+self.offset))
        elif self.loss_type == 'invNLL':
            return np.mean(y*preds - np.log(preds+self.offset))

    def _fit_scipy(self, xdata, ydata):
        """fitting method I: use scipy"""
        res = scipy.optimize.minimize(lambda betas: self._obj(xdata, ydata, betas),
                                      x0=self._params.get('initial_val',0.1*np.ones(self.num_bases)+1e-4),
                                      method=self.method)
        setattr(self, 'coefs', res.x)

    def _fit_cvxpy(self, xdata, ydata):
        """fitting method II: use cvxpy which is more stable"""
        bases = self.get_bases(xdata)
        bases_stacked = np.column_stack(list(bases.values()))
        
        beta = cp.Variable(bases_stacked.shape[1])
        
        if self.basis_type == 'b-spline':
            constraints= [0.0000001<=beta, beta<=99999.9]
        else:
            constraints= [-99999.9<=beta, beta<=99999.9]
            
        ydata = ydata.squeeze(-1)
        if self.loss_type == 'MSE':
            cost = cp.sum_squares(bases_stacked @ beta - ydata)
            problem = cp.Problem(cp.Minimize(cost),constraints=constraints)
            result = problem.solve(solver=cp.ECOS, verbose=True, max_iters=999)
        elif self.loss_type == 'NLL':
            raise ValueError('not supported for using cvxpy')
            #cost = cp.sum(cp.multiply(ydata,cp.inv_pos(bases_stacked@beta)) + cp.log(bases_stacked@beta + self.offset))
        elif self.loss_type == 'invNLL':
            cost = cp.sum(cp.multiply(ydata,bases_stacked@beta + self.offset) - cp.log(bases_stacked@beta + self.offset))
            problem = cp.Problem(cp.Minimize(cost),constraints=constraints)
            result = problem.solve(solver=cp.ECOS, verbose=True, max_iters=999)
        setattr(self, 'coefs', beta.value)
        
    def fit(self, xdata, ydata):
    
        if self.optimizer_engine == 'cvxpy':
            self._fit_cvxpy(xdata, ydata)
        elif self.optimizer_engine == 'scipy':
            self._fit_scipy(xdata, ydata)
        else:
            raise ValueError('unrecognized optimizer_engine')

    def run_on_testset(self, xdata):
        vals = self.functional(xdata, self.coefs)
        if self.loss_type == 'invNLL':
            vals = 1./(self.offset+vals)
            
        vals = np.clip(vals, self.f_min, self.f_max)
        return vals
