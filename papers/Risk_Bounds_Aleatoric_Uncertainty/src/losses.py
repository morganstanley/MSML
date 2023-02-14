""" some customized loss class """

import torch
import torch.nn as nn

class NegLoglikLoss(nn.Module):
    """
    vanilla loss = log (sigma)^2 + [y - f(x)]^2 / (sigma)^2
    with fixed mean, using plug in,
    loss = log (sigma_hat^2) + sigma^2/(sigma_hat^2)
    """

    def __init__(self, is_modeling_inverse=False):
    
        super(NegLoglikLoss, self).__init__()
        self.is_modeling_inverse = is_modeling_inverse

    def nll_fn(self, preds, targets):
        loss = torch.mean(torch.div(targets,preds) + torch.log(preds))
        return loss
    
    def inv_nll_fn(self, preds, targets):
        loss = torch.mean(targets * preds - torch.log(preds))
        return loss
        
    def forward(self, preds, targets):
        if self.is_modeling_inverse:
            return self.inv_nll_fn(preds, targets)
        else:
            return self.nll_fn(preds, targets)

class StaticComboLoss(NegLoglikLoss):
    """ combo loss with static weight: w_MSE * MSE loss + w_NLL * NLL loss """
    def __init__(
        self,
        is_modeling_inverse=False,
        w_MSE = None,
        w_NLL = None,
        w_min = 0.05,
    ):
        super().__init__(is_modeling_inverse)
        self.w_MSE = w_MSE
        self.w_NLL = w_NLL
        self.w_min = w_min
        self.mse_fn = nn.MSELoss()
    
    def forward(self, preds, targets):
        if not self.is_modeling_inverse:
            loss_from_mse = self.mse_fn(preds, targets)
            loss_from_nll = self.nll_fn(preds, targets)
        else:
            loss_from_mse = self.mse_fn(torch.div(1, preds), targets)
            loss_from_nll = self.inv_nll_fn(preds, targets)
            
        if self.w_MSE is None and self.w_NLL is None:
            normalizer = loss_from_mse + loss_from_nll
            w_MSE, w_NLL = loss_from_nll/normalizer, loss_from_mse/normalizer
            
            w_MSE, w_NLL = w_MSE.detach().cpu(), w_NLL.detach().cpu()
            if w_MSE <= w_NLL:
                w_MSE = max(self.w_min, w_MSE)
                w_NLL = 1 - w_MSE
            else:
                w_NLL = max(self.w_min, w_NLL)
                w_MSE = 1 - w_NLL
            setattr(self, 'w_MSE', w_MSE)
            setattr(self, 'w_NLL', w_NLL)
        return self.w_MSE * loss_from_mse + self.w_NLL * loss_from_nll

class StaticComboLogMSELoss(NegLoglikLoss):
    """
    NLL loss + log (MSE loss) with static weight
    """
    def __init__(
        self,
        is_modeling_inverse=False,
        w_MSE = None,
        w_NLL = None,
        w_min = 0.05,
    ):
        super().__init__(is_modeling_inverse)
        self.w_MSE = w_MSE
        self.w_NLL = w_NLL
        self.w_min = w_min
        self.mse_fn = nn.MSELoss()
    
    def forward(self, preds, targets):
        if not self.is_modeling_inverse:
            loss_from_mse = torch.log(1+self.mse_fn(preds, targets))
            loss_from_nll = self.nll_fn(preds, targets)
        else:
            loss_from_mse = torch.log(1+self.mse_fn(torch.div(1, preds), targets))
            loss_from_nll = self.inv_nll_fn(preds, targets)
        
        if self.w_MSE is None and self.w_NLL is None:
            normalizer = loss_from_mse + loss_from_nll
            w_MSE, w_NLL = loss_from_nll/normalizer, loss_from_mse/normalizer
            
            w_MSE, w_NLL = w_MSE.detach().cpu(), w_NLL.detach().cpu()
            if w_MSE <= w_NLL:
                w_MSE = max(self.w_min, w_MSE)
                w_NLL = 1 - w_MSE
            else:
                w_NLL = max(self.w_min, w_NLL)
                w_MSE = 1 - w_NLL
            setattr(self, 'w_MSE', w_MSE)
            setattr(self, 'w_NLL', w_NLL)
        
        return self.w_MSE * loss_from_mse + self.w_NLL * loss_from_nll
