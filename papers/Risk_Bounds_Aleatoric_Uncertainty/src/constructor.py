
import torch.nn as nn
from src import MLP, FCResNet
from src import NegLoglikLoss

class ModelCtor():
    def __init__(self,model_class,params):
        self.model_class = model_class
        self.params = params
        
    def ctor(self):
        if self.model_class == 'MLP':
            model = MLP(input_dim = self.params.get('input_dim',1)*self.params.get('n_degree',1),
                        hidden_dim = self.params['hidden_dim'],
                        output_dim = self.params.get('output_dim',1),
                        dropout_rate = self.params['dropout_rate'],
                        activation = self.params['activation'],
                        use_softplus = self.params['use_softplus'],
                        clip = self.params['clip'],
                        min_val = self.params.get('min_val',0.00001),
                        max_val = self.params.get('max_val',10000))
        elif self.model_class == 'FCResNet':
            model = FCResNet(input_dim = self.params.get('input_dim',1)*self.params.get('n_degree',1),
                             hidden_dim = self.params['hidden_dim'],
                             output_dim = self.params.get('output_dim',1),
                             dropout_rate = self.params['dropout_rate'],
                             activation = self.params['activation'],
                             architect = self.params['architect'],
                             use_softplus = self.params['use_softplus'],
                             clip = self.params['clip'],
                             min_val = self.params.get('min_val',0.00001),
                             max_val = self.params.get('max_val',10000))
        else:
            raise ValueError('currently not supported')

        return model
    
class LossCtor():
    def __init__(
        self,
        loss_type,
        params
    ):
        assert loss_type in ['MSE','NLL']
        self.loss_type = loss_type
        self.is_modeling_inverse = params.get('is_modeling_inverse', False)
        
    def ctor(self):
        if self.loss_type == 'MSE':
            loss_fn = nn.MSELoss()
        elif self.loss_type == 'NLL':
            loss_fn = NegLoglikLoss(self.is_modeling_inverse)
        return loss_fn
            
    

