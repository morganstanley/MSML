
import torch
import util

from ipdb import set_trace as debug

def build(opt, dyn, direction):
    print(util.magenta("build {} policy...".format(direction)))

    net_name = getattr(opt, direction+'_net')
    net = _build_net(opt, net_name)
    use_t_idx = (net_name in ['toy', 'Linear', 'Unet', 'DGLSB']) # t_idx is handled internally in ncsnpp
    scale_by_g = (net_name in ['ncsnpp'])

    policy = SchrodingerBridgePolicy(
        opt, direction, dyn, net, use_t_idx=use_t_idx, scale_by_g=scale_by_g
    )

    print(util.red('number of parameters is {}'.format(util.count_parameters(policy))))
    policy.to(opt.device)
    return policy

def _build_net(opt, net_name):
    zero_out_last_layer = opt.DSM_warmup

    if net_name == 'toy':
        assert util.is_toy_dataset(opt)
        if opt.diffusion == 'CLD':
            from models.toy_model.Toy_CLD import build_toy
            net = build_toy(opt.data_dim[0]*2, zero_out_last_layer, device=opt.device, num_ResNet=2)
        else:
            from models.toy_model.Toy_LD import build_toy
            net = build_toy(zero_out_last_layer)
    elif net_name.startswith('Linear') and 'Img' not in net_name:
        if opt.diffusion == 'CLD':
            from models.toy_model.Linear_CLD import LinearPolicy
            net = LinearPolicy(data_dim=opt.data_dim[0]*2, net_name=net_name, \
                               gamma=opt.gamma, damp_ratio=opt.damp_ratio)
        else:
            from models.toy_model.Linear_LD import LinearPolicy
            net = LinearPolicy(net_name=net_name)
    else:
        raise RuntimeError()
    return net

class SchrodingerBridgePolicy(torch.nn.Module):
    # note: scale_by_g matters only for pre-trained model
    def __init__(self, opt, direction, dyn, net, use_t_idx=False, scale_by_g=True):
        super(SchrodingerBridgePolicy,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx
        self.scale_by_g = scale_by_g

        self.beta_min = opt.beta_min
        self.beta_max = opt.beta_max
        self.beta_r = opt.beta_r
        self.interval = opt.interval

    @property
    def zero_out_last_layer(self):
        return self.net.zero_out_last_layer


    def forward(self, a, t):
        # make sure t.shape = [batch]
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(a.shape[0])
        assert t.dim()==1 and t.shape[0] == a.shape[0]

        if self.use_t_idx:
            t = t / self.opt.T * self.opt.interval

        if self.direction == 'forward' and self.opt.forward_net.startswith('Linear'):
            out = self.net(a, t, self.beta_min, self.beta_max, self.beta_r, self.interval, self.opt.baseline)
        else:
            out = self.net(a, t)

        # if the SB policy behaves as "Z" in FBSDE system,
        # the output should be scaled by the diffusion coefficient "g".
        if self.scale_by_g:
            g = self.dyn.g(t)
            g = g.reshape(a.shape[0], *([1,]*(a.dim()-1)))
            out = out * g

        return out


