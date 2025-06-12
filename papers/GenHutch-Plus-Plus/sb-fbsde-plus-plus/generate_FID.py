import os
import argparse

from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.utils as tu

import util
from ipdb import set_trace as debug

parser = argparse.ArgumentParser()
parser.add_argument("--problem-name")
parser.add_argument("--sde-type", type=str, default='ve', choices=['ve', 'vp'])
parser.add_argument("--FID-type", type=str, default='jpg', choices=['jpg', 'png'],   help="choose which type of FID to eval")

def generate_cifar10_dataset(opt, load_train=True):
    transforms_list=[
        transforms.Resize(32),
        transforms.ToTensor(), #Convert to [0,1]
    ]
    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.CIFAR10(
        'data',
        train=load_train,
        download=load_train,
        transform=transforms.Compose(transforms_list)
    )

def generate_celebA32_dataset(opt):
    transforms_list=[
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ]
    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.ImageFolder(
        root='data/celebA/img_align_celeba/',
        transform=transforms.Compose(transforms_list)
    )

def generate_celebA64_dataset(opt):
    transforms_list=[ #Normal Data preprocessing
        transforms.Resize([64,64]), #DSB type resizing
        transforms.ToTensor(),
    ]
    if util.use_vp_sde(opt):
        transforms_list+=[transforms.Lambda(lambda t: (t * 2) - 1),]

    return datasets.ImageFolder(
        root='data/celebA/img_align_celeba/',
        transform=transforms.Compose(transforms_list)
    )

def save_img_png_jpg(dataset, name, type):
    data_path='data/'+name+'-test/'
    os.makedirs(data_path, exist_ok=True)
    for i in range(len(dataset)):
        data,_=dataset[i]
        tu.save_image(data, data_path+'img{}.{}'.format(i,type))
        print(i)
    return data_path

opt = parser.parse_args()
FID_ref_name = input('Naming your FID reference:')

name, dataset_generator = {
    'celebA32':   ['celebA32', generate_celebA32_dataset],
    'celebA64':   ['celebA64', generate_celebA64_dataset],
    'cifar10':    ['cifar10',  generate_cifar10_dataset],
}.get(opt.problem_name)

dataset = dataset_generator(opt)
sample_path = save_img_png_jpg(dataset, name, opt.FID_type)
root='checkpoint/{}/'.format(opt.problem_name)

util.calculate_fid_npz(sample_path, root, FID_ref_name)
