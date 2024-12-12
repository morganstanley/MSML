"""
- NLL: NegLogLik, jointly optimize w.r.t. mean and variance
- sigmaNLL: optimize w.r.t. variance
- weightMSE: optimize w.r.t. mean
"""

import torch
import torch.nn as nn

class NegLoglikLoss(nn.Module):
    """
    NLL loss = log (sigma^2) + [y-f(x)]^2 / (sigma^2)
    where f(x) == preds_mean, sigma^2 == preds_var, y == target
    """
    def __init__(self, params):
    
        super(NegLoglikLoss, self).__init__()
        self.is_modeling_inverse = params['is_modeling_inverse']
        self.var_eps = params.get('var_eps',1e-6)
        
    def nll_fn(self, preds_mean, preds_var, targets):
        res_sq = (targets - preds_mean)**2
        preds_var = torch.clamp(preds_var, min=self.var_eps)
        loss = torch.mean(torch.div(res_sq,preds_var) + torch.log(preds_var))
        return loss
    
    def inv_nll_fn(self, preds_mean, preds_var, targets):
        res_sq = (targets - preds_mean)**2
        preds_var = torch.clamp(preds_var, max=1./self.var_eps)
        loss = torch.mean(res_sq * preds_var - torch.log(preds_var))
        return loss
    
    def forward(self, preds_mean, targets, preds_var):
        if self.is_modeling_inverse:
            return self.inv_nll_fn(preds_mean, preds_var, targets)
        else:
            return self.nll_fn(preds_mean, preds_var, targets)

class sigmaNLL(nn.Module):
    """
    use NLL loss for estimating sigma, while fixing the mean
    NLL loss = log (sigma)^2 + [y - f(x)]^2 / (sigma)^2
    In the case where the mean is fixed, one treats the squared residual as if it were the truth
    loss = log (sigma_hat^2) + sigma^2/(sigma_hat^2),
    where sigma^2 is the "target" value and sigma_hat^2 is the "predicted
    """
    def __init__(self, params):
    
        super(sigmaNLL, self).__init__()
        self.is_modeling_inverse = params['is_modeling_inverse']

    def nll_fn(self, preds, targets):
        loss = torch.mean(torch.div(targets,preds) + torch.log(preds))
        return loss
    
    def inv_nll_fn(self, preds, targets):
        loss = torch.mean(targets * preds - torch.log(preds))
        return loss
        
    def forward(self, preds, targets, *args):
        if self.is_modeling_inverse:
            return self.inv_nll_fn(preds, targets)
        else:
            return self.nll_fn(preds, targets)

class weightedMSE(nn.Module):
    """
    loss = weight * [y - f(x)]^2
    """
    def __init__(self, params=None):
    
        super(weightedMSE, self).__init__()
        self.mse_fn = nn.MSELoss(reduction='mean')
        self.apply_sigmoid = params.get('apply_sigmoid',False)
        
    def forward(self, preds, targets, weights=None):
    
        if self.apply_sigmoid:
            preds = torch.sigmoid(preds)
    
        if weights is None:
            return self.mse_fn(preds, targets)
        
        res_sq = (targets - preds)**2
        loss = torch.mean(res_sq * weights)
        
        return loss
        
class weightedBCE(nn.Module):
    
    def __init__(self, params=None):
    
        super(weightedBCE, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, preds, targets, weights=None):
        
        if weights is None:
            return torch.mean(self.bce(preds, targets))
        
        raw_bce = self.bce(preds, targets)
        loss = torch.mean(raw_bce * weights)
        
        return loss
