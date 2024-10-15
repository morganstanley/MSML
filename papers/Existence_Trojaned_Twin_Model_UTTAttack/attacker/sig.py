from collections import defaultdict

import numpy as np
import cv2

from .attacker import Attacker

class SIG(Attacker):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int, 
                     xi: float, 
                     **kwargs) -> np.ndarray:
        
        return (1-self.lamda)*img + self.lamda*xi*self.trigger[label]
    
    def _generate_trigger(self) -> np.ndarray:
        
        imgshape = self.config['dataset'][self.argsdataset]['IMG_SIZE']
        self.trigger = defaultdict(np.ndarray)
        img_size = int(self.config['dataset'][self.config['args']['dataset']]['IMG_SIZE'])
        for k in self.target_source_pair:
            self.trigger[k] =  20*np.sin(2*np.pi*6*np.linspace(1, img_size, img_size)/img_size)
            self.trigger[k] =  np.repeat(cv2.resize(self.trigger[k][None, :], (imgshape, imgshape))[:, :, None], axis=2, repeats=3)
            self.trigger[k] *= self.budget/(np.linalg.norm(self.trigger[k].reshape(3, -1), ord='fro')+1e-4) #L2 norm constrain
