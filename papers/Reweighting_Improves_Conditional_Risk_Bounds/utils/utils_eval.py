import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def get_threshold(abst_lvl, ref_vals, mode='upper'):
    
    assert mode in ['upper','lower']
    if mode == 'upper':
        return np.quantile(ref_vals, 1-abst_lvl)
    else:
        return np.quantile(ref_vals, abst_lvl)
        
def get_selective_set(hats, threshold, mode='leq'):
    
    assert mode in ['leq','geq']
    if mode == 'leq':
        return np.where(hats <= threshold)[0]
    else:
        return np.where(hats >= threshold)[0]
        

def risk_selectset(y_true, y_pred, selective_set=None, run_type='regr'):
    
    assert run_type in ['regr','clsf']
    if selective_set is not None:
        y_true, y_pred = y_true[selective_set], y_pred[selective_set]
    
    if run_type == 'regr':
        return mean_squared_error(y_true, y_pred)
    else:
        y_pred = 1 * (y_pred >= 0.5)
        return 1 - accuracy_score(y_true, y_pred)
        
