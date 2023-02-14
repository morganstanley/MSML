
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class Evaluator():
    
    def __init__(self):
        self.metrics = ['rmse','mae','ratio']
    
    def rmse(self, est, truth):
        return mean_squared_error(truth, est, squared=False)
    def mae(self, est, truth):
        return mean_absolute_error(truth, est)
    def ratio(self, est, truth):
        return np.mean(est/truth)
        
    def report(self, est, truth):
        report_container = {}
        for metric in self.metrics:
            fn = getattr(self, metric)
            report_container[metric] = fn(est, truth)
        return report_container
        
        
        
        
