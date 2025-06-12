import numpy as np
import torch
import torch.distributions as td
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from prefetch_generator import BackgroundGenerator

import util
from ipdb import set_trace as debug

def build_boundary_distribution(opt, Cov=None):
    print(util.magenta("build boundary distribution..."))
    opt.data_dim = get_data_dim(opt.problem_name)
    prior = build_prior_sampler(opt, opt.samp_bs, Cov)
    pdata = build_data_sampler(opt, opt.samp_bs)

    return pdata, prior

def get_data_dim(problem_name):
    return {
        'gaussian':     [2],
        'gmm':          [2],
        'checkerboard': [2],
        'spiral':        [2],
        'mnist':       [1,32,32],
        'celebA32':    [3,32,32],
        'celebA64':    [3,64,64],
        'cifar10':     [3,32,32],
    }.get(problem_name)

def build_prior_sampler(opt, batch_size, Cov, means=None):
    if Cov == None:
        Cov = torch.eye(opt.data_dim[-1])
    if means == None:
        means = torch.zeros(opt.data_dim)
    prior = td.MultivariateNormal(means, Cov)
    return PriorSampler(prior, batch_size, opt.device)

def build_data_sampler(opt, batch_size):
    if util.is_toy_dataset(opt):
        if opt.problem_name == 'gaussian':
            means = torch.Tensor([-8, 2])
            Covariance = torch.Tensor([[1., 1.], 
                                       [1., 5.]])
            return build_prior_sampler(opt, batch_size, Covariance, means)
        else:
            return {
                'gmm': MixMultiVariateNormal,
                'checkerboard': CheckerBoard,
                'spiral': Spiral,
            }.get(opt.problem_name)(batch_size, opt.x_scalar, opt.y_scalar)

    elif util.is_image_dataset(opt):
        dataset_generator = {
            'mnist':      generate_mnist_dataset,
            'celebA32':   generate_celebA_dataset,
            'celebA64':   generate_celebA_dataset,
            'cifar10':    generate_cifar10_dataset,
        }.get(opt.problem_name)
        dataset = dataset_generator(opt)
        return DataSampler(dataset, batch_size, opt.device)

    else:
        raise RuntimeError()

class MixMultiVariateNormal:
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0, radius=12, num=8, sigmas=None):
        # build mu's and sigma's
        arc = 2*np.pi/num
        xs = [np.cos(arc*idx)*radius for idx in range(num)]
        ys = [np.sin(arc*idx)*radius for idx in range(num)]
        mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]
        dim = len(mus[0])
        sigmas = [torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self,x):
        # assume equally-weighted
        densities=[torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities)/len(self.dists))

    def sample(self):
        ind_sample = self.batch_size/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        samples[:, 0] = self.x_scalar * samples[:, 0]
        samples[:, 1] = self.y_scalar * samples[:, 1]
        return samples

class CheckerBoard:
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0):
        self.batch_size = batch_size
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar

        print(util.red(f'rescale data with {self.x_scalar}x on X-axis  and {self.y_scalar}x Y-axis.'))

    def sample(self):
        n = self.batch_size
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

        sample[:, 0] = self.x_scalar * sample[:, 0]
        sample[:, 1] = self.y_scalar * sample[:, 1]
        return sample

class Spiral:
    def __init__(self, batch_size, x_scalar=1.0, y_scalar=1.0):
        self.batch_size = batch_size
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar

        print(util.red(f'rescale data with {self.x_scalar}x on X-axis  and {self.y_scalar}x Y-axis.'))

    def sample(self):
        n = self.batch_size
        theta = np.sqrt(np.random.rand(n))*3*np.pi-0.5*np.pi # np.linspace(0,2*pi,100)

        r_a = theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 0.25*np.random.randn(n,2)
        samples = np.append(x_a, np.zeros((n,1)), axis=1)
        samples = samples[:,0:2] * 0.4 # default 0.4 re-scaled
        samples[:, 0] = self.x_scalar * samples[:, 0]
        samples[:, 1] = self.y_scalar * samples[:, 1]
        return torch.Tensor(samples)


class DataSampler: # a dump data sampler
    def __init__(self, dataset, batch_size, device):
        self.num_sample = len(dataset)
        self.dataloader = setup_loader(dataset, batch_size, device)
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        data = next(self.dataloader)
        return data[0].to(self.device)

class PriorSampler: # a dump prior sampler to align with DataSampler
    def __init__(self, prior, batch_size, device):
        self.prior = prior
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def sample(self):
        return self.prior.sample([self.batch_size]).to(self.device)

def setup_loader(dataset, batch_size, device):
    train_loader = DataLoaderX(dataset, batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True, generator=torch.Generator(device=device))
    # train_loader = DataLoaderX(dataset, batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True)
    print("number of samples: {}".format(len(dataset)))

    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py#L52-L53
    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/train_util.py#L166
    while True:
        yield from train_loader

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
        transforms.Pad(2,fill=0), #left and right 2+2=4 padding
        transforms.ToTensor(),
    ]
    transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.MNIST(
        'data',
        train= not opt.compute_NLL,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )

def generate_cifar10_dataset(opt,load_train=True):
    transforms_list=[
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), #Convert to [0,1]
    ]
    transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.CIFAR10(
        'data',
        train= not opt.compute_NLL,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )
