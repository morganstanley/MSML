import numpy as np

import torch
import torch.distributions as td
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from ipdb import set_trace as debug

from scipy.interpolate import InterpolatedUnivariateSpline
from prefetch_generator import BackgroundGenerator
import util

NUM_WORKERS = 0

def build_boundary_distribution(opt):
    print(util.magenta("build boundary distribution..."))

    opt.data_dim = get_data_dim(opt)
    prior = build_prior_sampler(opt, opt.samp_bs)
    pdata = build_data_sampler(opt, opt.samp_bs)

    return pdata, prior

def get_data_dim(opt):
    problem_name = opt.problem_name
    if problem_name in ['sinusoid', 'sinusoid_large', 'pm25', 'physio', 'tba2017', 'etf052023',
        'exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
        return [1, opt.input_size[0], opt.input_size[1]]
    elif problem_name in ['gmm', 'checkerboard', 'moon-to-spiral', 'gaussian-to-gaussian',
        'mnist', 'celebA32', 'celebA64', 'cifar10', 'moon', 'spiral']:
        return {
            'gmm':            [2],
            'checkerboard':   [2],
            'moon-to-spiral': [2],
            'moon':           [2],
            'spiral':         [2],
            'gaussian-to-gaussian': [2],
            'mnist':          [1,32,32],
            'celebA32':       [3,32,32],
            'celebA64':       [3,64,64],
            'cifar10':        [3,32,32],
        }.get(problem_name)
    elif problem_name in ['inception']:
        return {
            'inception':      [1008],
        }.get(problem_name)
    else:
        raise NotImplementedError('New dataset type.')

def build_prior_sampler(opt, batch_size):

    if opt.uniform_prior:
        print(util.green('Uniform prior'))
        prior = td.Uniform(low=torch.zeros(opt.data_dim, device=opt.device),
                           high=torch.ones(opt.data_dim, device=opt.device))
        return PriorSampler(prior, batch_size, opt.device)

    if opt.problem_name == 'moon-to-spiral':
        # 'moon-to-spiral' uses Moon as prior distribution
        return Moon(batch_size)

    elif opt.problem_name == 'gaussian-to-gaussian':
        # mu = torch.zeros(2, device=opt.device)
        # sigma = torch.eye(2, device=opt.device) * 1
        mu = torch.Tensor([-5, -2])
        sigma = torch.tensor([[2., -1.5], [-1.5, 2.0]])
        prior = td.MultivariateNormal(mu, sigma)
        return PriorSampler(prior, batch_size, opt.device)

    # image+VESDE -> use (sigma_max)^2; otherwise use 1.
    # cov_coef = opt.sigma_max**2 if (util.is_image_dataset(opt) and not util.use_vp_sde(opt)) else 1.
    if util.is_image_dataset(opt) and not util.use_vp_sde(opt):
        cov_coef = opt.sigma_max**2
    elif opt.sde_type == 've' and opt.problem_name == 'gmm':
        cov_coef = opt.sigma_max**2 + 7**2  # mimic the forward marginal prior distribution.
    elif opt.sde_type in ['ve', 'rve']:
        cov_coef = opt.sigma_max**2
    else:
        cov_coef = 1  # vpsde. OU process, converge to unit Gaussian.

    if hasattr(opt, 'prior_sigma') and opt.prior_sigma is not None:
        cov_coef = opt.prior_sigma**2
        print(util.yellow(f'Overwrite prior sigma due to opt.prior_sigma'))

    print(util.green('prior cov'), cov_coef)
    prior = td.MultivariateNormal(torch.zeros(opt.data_dim, device=opt.device),
                                  cov_coef*torch.eye(opt.data_dim[-1], device=opt.device))
    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt, batch_size):
    if util.is_toy_dataset(opt):
        return {
            'gmm': MixMultiVariateNormal,
            'gaussian-to-gaussian': Gaussian,
            'checkerboard': CheckerBoard,
            'moon-to-spiral': Spiral,
            'moon': Moon,
            'spiral': Spiral,
        }.get(opt.problem_name)(batch_size, device=opt.device)

    elif util.is_image_dataset(opt):
        dataset_generator = {
            'mnist':      generate_mnist_dataset,
            'celebA32':   generate_celebA_dataset,
            'celebA64':   generate_celebA_dataset,
            'cifar10':    generate_cifar10_dataset,
            'imagenet32': generate_imagenet32_dataset,
        }.get(opt.problem_name)
        dataset = dataset_generator(opt)
        return DataSampler(dataset, batch_size, opt.device, opt=opt)

    elif opt.problem_name in ['sinusoid', 'sinusoid_large']:
        dataset_generator = generate_sinusoid_dataset
        dataset = dataset_generator(opt)
        return DataSampler(dataset, batch_size, opt.device, opt=opt, torch_loader=False)

    elif opt.problem_name == 'pm25':
        dataset_generator = generate_pm25_dataset
        dataset = dataset_generator(opt)
        return DataSampler(dataset, batch_size, opt.device, opt=opt, torch_loader=False)

    elif opt.problem_name == 'physio':
        dataset_generator = generate_physio_dataset
        dataset = dataset_generator(opt)
        return DataSampler(dataset, batch_size, opt.device, opt=opt, torch_loader=False)

    elif opt.problem_name in ['exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
        return DataSampler(opt.problem_name, batch_size, opt.device, opt=opt, torch_loader=False)

    elif opt.problem_name in ['inception',]:
        dataset = generate_inception_dataset(opt)
        return DataSampler(dataset, batch_size, opt.device, opt=opt, torch_loader=False)

    else:
        raise NotImplementedError('New dataset type.')


# class MixMultiVariateNormal:
#     def __init__(self, batch_size, radius=12, num=8, sigmas=None):

#         # build mu's and sigma's
#         arc = 2*np.pi/num
#         xs = [np.cos(arc*idx)*radius for idx in range(num)]
#         ys = [np.sin(arc*idx)*radius for idx in range(num)]
#         mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
#         dim = len(mus[0])
#         sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

#         if batch_size%num!=0:
#             raise ValueError('batch size must be devided by number of gaussian')
#         self.num = num
#         self.batch_size = batch_size
#         self.dists=[
#             td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
#         ]

#     def log_prob(self,x):
#         # assume equally-weighted
#         densities=[torch.exp(dist.log_prob(x)) for dist in self.dists]
#         return torch.log(sum(densities)/len(self.dists))

#     def sample(self):
#         ind_sample = self.batch_size/self.num
#         samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
#         samples=torch.cat(samples,dim=0)
#         return samples


class MixMultiVariateNormal:
    def __init__(self, batch_size, radius=12, num=7, sigmas=None, device='cpu'):
        # build mu's and sigma's. num: number of modes.
        arc = 2*np.pi/num
        xs = [np.cos(arc*idx)*radius for idx in range(num)]
        ys = [np.sin(arc*idx)*radius for idx in range(num)]
        # mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        # dim = len(mus[0])
        # sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)] + [torch.Tensor([0,0])]
        dim = len(mus[0])
        sigmas = [torch.eye(dim)*2.2 for _ in range(num+1)] if sigmas is None else sigmas

        print('batch_size', batch_size, 'num', num)
        if batch_size%(num+1)!=0:
            raise ValueError(f'batch size {batch_size} must be devided by number of gaussian {num}')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]
        self.device = device

    def log_prob(self,x):
        # assume equally-weighted
        densities=[torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities)/len(self.dists))

    def sample(self, num_samples=None):
        if num_samples is None:
            num_samples = self.batch_size
        # build mu's and sigma's. num: number of modes.
        ind_sample = num_samples / (self.num+1)
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        return samples.to(self.device)

class CheckerBoard:
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device

    def sample(self, num_samples=None):
        n = self.batch_size if num_samples is None else num_samples
        n_points = 3*n
        n_classes = 2
        freq = 5
        x = np.random.uniform(-(freq//2)*np.pi, (freq//2)*np.pi, size=(n_points, n_classes))
        mask = np.logical_or(np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0), \
        np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0))
        y = np.eye(n_classes)[1*mask]
        x0=x[:,0]*y[:,0]
        x1=x[:,1]*y[:,0]
        sample=np.concatenate([x0[...,None],x1[...,None]],axis=-1)
        sqr=np.sum(np.square(sample),axis=-1)
        idxs=np.where(sqr==0)
        sample=np.delete(sample,idxs,axis=0)
        # res=res+np.random.randn(*res.shape)*1
        sample=torch.Tensor(sample)
        sample=sample[0:n,:]
        return sample

class Spiral:
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        self.num_sample = 100000 # Fake number.

    def sample(self, num_samples=None):
        n = self.batch_size if num_samples is None else num_samples
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n,2)
        samples = np.append(x_a, np.zeros((n,1)), axis=1)
        samples = samples[:,0:2]/25 + 0.45  # Shift to [0,1]^d domain.
        return torch.Tensor(samples).to(self.device)

class Moon:
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device

    def sample(self, num_samples=None):
        n = self.batch_size if num_samples is None else num_samples
        x = np.linspace(0, np.pi, n // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1) * 10.
        u += 0.5*np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1) * 10.
        v += 0.5*np.random.normal(size=v.shape)
        x = np.concatenate([u, v], axis=0)
        x = x + 0.25*np.random.randn(n,2)
        x = x/45 + 0.5  # Shift to [0,1]^d domain.
        return torch.Tensor(x).to(self.device)


class Gaussian:
    def __init__(self, batch_size, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        mu = torch.Tensor([12, 12])
        sigma = torch.tensor([[10., 6.], [6., 4]])
        self.dist = td.multivariate_normal.MultivariateNormal(mu, sigma)

    def sample(self, num_samples=None):
        n = self.batch_size if num_samples is None else num_samples
        samples=self.dist.sample([n])
        return torch.Tensor(samples).to(self.device)


class DataSampler: # a dump data sampler
    def __init__(self, dataset=None, batch_size=32, device='cpu', opt=None, torch_loader=False):
        self.num_sample = len(dataset)

        if opt.problem_name in ['exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
            self.dataloader = setup_loader_nips(opt.problem_name, batch_size, opt=opt)
        else:
            self.dataloader = setup_loader(dataset, batch_size, torch_loader=torch_loader)
        self.batch_size = batch_size
        self.device = device
        self.opt = opt

    def sample(self, num_samples=None, return_mask=False, return_all_mask=False):
        # need batch_permute
        if self.opt.problem_name in ['sinusoid', 'sinusoid_large', 'pm25', 'physio',
            'exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
            assert self.opt.permute_batch == True

            data = next(iter(self.dataloader))
            obs_data = data['observed_data'].float().permute(0,2,1)  # (B,L,K).permute->(B,K,L)

            if return_all_mask:
                obs_mask = data['observed_mask'].float().permute(0,2,1)  # (B,L,K) (B,K,L)
                obs_data = obs_data * obs_mask
                gt_mask = data["gt_mask"].float().permute(0,2,1)
                return (obs_data.unsqueeze(1).to(self.device),
                        obs_mask.unsqueeze(1).to(self.device),
                        gt_mask.unsqueeze(1).to(self.device))

            elif return_mask:
                obs_mask = data['observed_mask'].float().permute(0,2,1)  # (B,L,K) (B,K,L)
                obs_data = obs_data * obs_mask
                # if hasattr(self.opt, 'interpolate') and self.opt.interpolate:
                #     obs_data = interpolate(obs_data, obs_mask)  # x, mask (B,K,L)
                return (obs_data.unsqueeze(1).to(self.device),
                        obs_mask.unsqueeze(1).to(self.device))
            else:
                return obs_data.unsqueeze(1).to(self.device)

        elif self.opt.problem_name in ['gmm', 'checkerboard', 'moon-to-spiral', 'gaussian-to-gaussian',
            'cifar10', 'celebA64', 'celebA32', 'mnist', 'moon', 'spiral',]:
            # data = next(self.dataloader)  # original code using yield.
            data = next(iter(self.dataloader))
            return data[0].to(self.device)
        elif self.opt.problem_name in ['inception',]:
            data = next(iter(self.dataloader))
            return data.to(self.device)
        else:
            raise NotImplementedError(f'Dataset {self.opt.problem_name} is unknown.')


def interpolate(observed_data, cond_mask):
    B, K, L = observed_data.shape
    cond_obs = cond_mask * observed_data
    timeline = np.arange(L)
    cond_mask = cond_mask.cpu().numpy()
    imputed_samples = cond_obs.clone().cpu().numpy()
    interpolate_order = 1

    for b in range(B):
        for k in range(K):
            obs_points = cond_mask[b,k] == 1
            target_points = ~obs_points
            if np.sum(obs_points) == L:
                continue
            obs_x = timeline[obs_points]
            obs_y = observed_data[b,k,obs_points].cpu().numpy()
            if np.sum(obs_points) > interpolate_order:  # Empty feature.
                target_x = timeline[target_points]
                target_x_clipped = np.clip(target_x, a_min=obs_x[0], a_max=obs_x[-1])
                interpolate_func = InterpolatedUnivariateSpline(obs_x, obs_y, k=interpolate_order)
                inpute_y = interpolate_func(target_x_clipped)
            else:
                num_obs = cond_mask[b].sum(axis=0)  # L
                num_obs[num_obs == 0] = 1
                obs_y_mean = cond_obs[b].cpu().numpy().sum(axis=0) / num_obs  # L, Mean over K.
                inpute_y = obs_y_mean[target_points]

            imputed_samples[b,k,target_points] = inpute_y

    return torch.from_numpy(imputed_samples)

class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device):
        self.prior = prior
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self, num_samples=None):
        num_samples = self.batch_size if num_samples is None else num_samples
        return self.prior.sample([num_samples]).to(self.device)

def setup_loader_nips(dataset: str, batch_size: int=32, opt=None):
    from dataset_nips import get_dataloader

    train_loader, val_loader, test_loader, scaler, mean_scaler = get_dataloader(
        dataset, batch_size=batch_size, device='cpu', target_dim_range=opt.target_dim_range)
    while True:
        yield from train_loader


def setup_loader(dataset=None, batch_size=32, torch_loader=False):

    if not torch_loader:
        # train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True,
        #      num_workers=NUM_WORKERS, drop_last=True)
        train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True)
        # train_loader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True,
        #     num_workers=NUM_WORKERS, pin_memory=True)
        # print("number of samples: {}".format(len(dataset)))
        # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py#L52-L53
        # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/train_util.py#L166
        while True:
            yield from train_loader
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=False)
        while True:
            yield from train_loader
        # return train_loader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def generate_celebA_dataset(opt,load_train=True):
    if opt.problem_name=='celebA32': #Our own data preprocessing
        transforms_list=[
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    elif opt.problem_name=='celebA64':
        transforms_list=[ #Normal Data preprocessing
            transforms.Resize([64,64]), #DSB type resizing
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    else:
        raise RuntimeError()

    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.ImageFolder(
        root='data/celebA/img_align_celeba/',
        transform=transforms.Compose(transforms_list)
    )

def generate_mnist_dataset(opt,load_train=True):
    transforms_list=[
        transforms.Pad(2, fill=0), #left and right 2+2=4 padding
        transforms.ToTensor(),
    ]

    # TEST
    print('TEST MNIST using rescale.')
    # transforms_list+=[transforms.Lambda(lambda t: (t / 2) + 0.25),]
    transforms_list+=[transforms.Lambda(lambda t: (t / 4) + 0.375),]

    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.MNIST(
        'data/mnist/',
        train= not opt.compute_NLL,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )

def generate_cifar10_dataset(opt, load_train=True):
    transforms_list=[
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), #Convert to [0,1]
    ]
    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.CIFAR10(
        'data/cifar10/',
        train= not opt.compute_NLL,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )

def generate_imagenet32_dataset(opt, load_train=True):
    transforms_list=[
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), #Convert to [0,1]
    ]
    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.CIFAR10(
        'data/cifar10/',
        train= not opt.compute_NLL,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )

def generate_inception_dataset(opt, load_train=True):
    datasets = np.load('data/imagenet-64x64/inception_probabilities/imagenet64c_valid_50k.npy')
    datasets = datasets.astype(np.float32)
    return datasets

def generate_sinusoid_dataset(opt):
    import dataset_sinusoid
    train_dataset, valid_dataset, test_dataset = dataset_sinusoid.get_dataloader(
        opt.sinusoid_dataset_path, batch_size=None, return_dataset=True, seed=1)
    print('num samples', len(train_dataset), len(valid_dataset), len(test_dataset))
    return train_dataset

def generate_pm25_dataset(opt):
    import dataset_pm25
    train_dataset, valid_dataset, test_dataset = dataset_pm25.get_dataloader(
        batch_size=None, eval_length=36, target_dim=36, validindex=0, device='cpu',
        return_dataset=True)
    print('num samples', len(train_dataset), len(valid_dataset), len(test_dataset))
    return train_dataset

def generate_physio_dataset(opt):
    import dataset_physio
    train_dataset, valid_dataset, test_dataset = dataset_physio.get_dataloader(
        seed=1, nfold=opt.physio_nfold, missing_ratio=opt.dataset_missing_ratio, batch_size=None,
        eval_length=48, target_dim=35, device='cpu', return_dataset=True)
    print('num samples', len(train_dataset), len(valid_dataset), len(test_dataset))
    return train_dataset

