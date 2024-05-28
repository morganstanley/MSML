from collections import defaultdict

import numpy as np

from .attacker import Attacker

class BadNet(Attacker):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.trigger_w = int(self.config['attack']['badnet']['TRIGGER_SHAPE'])

    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int, 
                     xi: float, 
                     **kwargs) -> np.ndarray:
        
        pos = np.random.choice(['topleft', 'topright', 'bottomleft', 'bottomright'], 1, replace=False)
        
        trigger_w = min(self.trigger_w, min(img.shape[0], img.shape[1]))
        if pos=='topleft':
            h_s, h_e = 0, trigger_w
            w_s, w_e = 0, trigger_w
        elif pos=='topright':
            h_s, h_e = img.shape[0]-trigger_w, img.shape[0]
            w_s, w_e = 0, trigger_w
        elif pos=='bottomleft':
            h_s, h_e = 0, trigger_w
            w_s, w_e = img.shape[1]-trigger_w, img.shape[1]
        else: # pos='bottomright'
            h_s, h_e = img.shape[0]-trigger_w, img.shape[0]
            w_s, w_e = img.shape[1]-trigger_w, img.shape[1]
        
        self.content = np.zeros(img.shape, dtype=np.float32)
        self.content[h_s:h_e, w_s:w_e] = self.trigger[label]

        return (1-self.lamda)*img + self.lamda*xi*self.content

    def _generate_trigger(self) -> None:
        # random pattern trigger
        self.trigger = defaultdict(np.ndarray)
        for k in self.config['attack']['SOURCE_TARGET_PAIR']:
            self.trigger[k] = np.random.uniform(0, 1, 3*self.trigger_w**2).reshape(self.trigger_w, self.trigger_w, 3)
            self.trigger[k] *= self.budget/(np.linalg.norm(self.trigger[k].reshape(3, -1), ord='fro')+1e-4) #L2 norm constrain
   