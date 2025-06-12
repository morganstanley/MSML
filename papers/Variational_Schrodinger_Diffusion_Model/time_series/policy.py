
from typing import Optional
from torch import nn, Tensor

from ipdb import set_trace as debug

def build(
    net,
    dyn,
    direction,
):
    policy = SchrodingerBridgePolicy(
        direction, dyn, net, use_t_idx=False, scale_by_g=False,
    )
    return policy


class SchrodingerBridgePolicy(nn.Module):
    # note: scale_by_g matters only for pre-trained model
    def __init__(self, direction, dyn, net, use_t_idx=False, scale_by_g=True):
        super(SchrodingerBridgePolicy,self).__init__()
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx
        self.scale_by_g = scale_by_g

    @ property
    def zero_out_last_layer(self):
        return self.net.zero_out_last_layer

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        # make sure t.shape = [batch]
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(x.shape[0])
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        out = self.net(x, t, cond=cond)

        # if the SB policy behaves as "Z" in FBSDE system,
        # the output should be scaled by the diffusion coefficient "g".
        if self.scale_by_g:
            g = self.dyn.g(t)
            g = g.reshape(x.shape[0], *([1,]*(x.dim()-1)))
            out = out * g

        return out
