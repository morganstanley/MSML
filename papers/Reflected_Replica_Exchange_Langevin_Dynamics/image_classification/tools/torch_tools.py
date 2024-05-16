import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from datasets import load_dataset
from torchvision import transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset


def loader(train_size, test_size, args):
    if args.data.startswith('cifar'):
        if args.data == 'cifar10':
            dataloader = datasets.CIFAR10
        else:
            dataloader = datasets.CIFAR100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.5),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        exit('Unknown dataset')

    trainset = dataloader('./data/' + args.data.upper(), train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=train_size, shuffle=True,
                                   num_workers=0)  # num_workers=0 is crucial for seed
    """ caution: no shuffle on test dataset """
    testset = dataloader(root='./data/' + args.data.upper(), train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def load_ImageNet(train_size, test_size, args, num_workers=4):
    ds = load_dataset("imagenet-1k")
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Define transformations
    transform = transforms.Compose([
        # ... other transforms ...
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class CustomImageNetDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset

            weights = ResNet50_Weights.DEFAULT
            transform = weights.transforms()

            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx]["image"]
            label = self.dataset[idx]["label"]

            if image.mode != 'RGB':
                image = image.convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

    # Wrap your dataset
    train_ds_wrapped = CustomImageNetDataset(train_ds, transform=transform)
    test_ds_wrapped = CustomImageNetDataset(test_ds, transform=transform)

    # Create the DataLoader
    train_loader = DataLoader(train_ds_wrapped, batch_size=train_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds_wrapped, batch_size=test_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def uncertainty_estimation(net, test_loader, extra_loader, prob_avg_seen, prob_avg_unseen, weight, acc_weights, counter,
                           print_tag=True, info='vanilla'):
    softmax = nn.Softmax(dim=1)
    for TT in prob_avg_seen:
        for cnt, (images, labels) in enumerate(extra_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob = softmax(net.forward(images).data / TT) * (weight + 1e-10)
            if counter == 1:
                prob_avg_unseen[TT].append(prob)
            else:
                prob_avg_unseen[TT][cnt] += prob

        Brier_unseen = 0
        counts_unseen = 0
        for cnt, (images, labels) in enumerate(extra_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            prob_unseen = prob_avg_unseen[TT][cnt] * 1. / acc_weights
            counts_unseen += prob_unseen.shape[0]
            Brier_unseen += torch.mean((prob_unseen) ** 2, dim=1).sum().item()

        if print_tag == True:
            print('\n' + '===' * 50)
            print('{} scaling {} Unseen  / {:.5f}'.format(info, TT, Brier_unseen / counts_unseen))


def compute_brier_score(probabilities, labels):
    # Assumption: labels are one-hot encoded
    # probabilities are the predicted probabilities for each class
    # Brier score is calculated as the mean squared error between the predicted probabilities and the actual outcomes
    brier_scores = np.mean(np.sum((probabilities - labels) ** 2, axis=1))
    return brier_scores


class BayesEval:
    def __init__(self, data='cifar100'):
        self.counter = 1
        self.bma = []
        self.cur_acc = 0
        self.bma_acc = 0
        self.best_cur_acc = 0
        self.best_bma_acc = 0
        self.best_nll = float('inf')
        self.best_bma_nll = float('inf')
        self.acc_weights = 0
        self.brier_score = 0
        self.best_brier = float('inf')

        self.prob_avg_seen = {1: []} if data == 'cifar100' else {1: [], 2: []}
        self.prob_avg_unseen = {1: []} if data == 'cifar100' else {1: [], 2: []}

    def eval(self, net, test_loader, extra_loader, criterion, weight=1, bma=False, iters=None):
        net.eval()
        one_correct, bma_correct, self.nll, self.bma_nll = 0, 0, 0, 0
        """ Non-convex optimization """
        total_brier_score, total_samples = 0, 0

        for cnt, (images, labels) in enumerate(test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net.forward(images).data

            self.nll += (criterion(outputs, labels) * outputs.shape[0]).item()
            one_correct += outputs.max(1)[1].eq(labels.data).sum().item()
            if bma == True:
                outputs = outputs * (weight + 1e-10)
                if cnt == 0:
                    self.acc_weights += (weight + 1e-10)
                if self.counter == 1:
                    self.bma.append(outputs)
                else:
                    self.bma[cnt] += outputs
                bma_correct += self.bma[cnt].max(1)[1].eq(labels.data).sum().item()
                self.bma_nll += (criterion(self.bma[cnt] * 1. / self.acc_weights, labels) * outputs.shape[0]).item()

            # For Brier score
            probabilities = self.bma[cnt].softmax(dim=1).cpu().numpy()  # Convert logits to probabilities
            true_labels = labels.data.cpu().numpy()
            one_hot_labels = np.eye(probabilities.shape[1])[true_labels]

            # Calculate and accumulate Brier score
            total_brier_score += compute_brier_score(probabilities, one_hot_labels)
            total_samples += images.size(0)

        self.brier_score = total_brier_score / total_samples
        if iters is not None and iters >= 20:
            self.best_brier = min(self.brier_score, self.best_brier)

        if bma == True:
            uncertainty_estimation(
                net, test_loader, extra_loader, self.prob_avg_seen,
                self.prob_avg_unseen, weight, self.acc_weights, self.counter)
            # uncertainty_estimation(data, net, test_loader, extra_loader, self.prob_avg_seen, self.prob_avg_unseen, weight, self.acc_weights, self.counter)
            self.counter += 1

        self.cur_acc = 100.0 * one_correct / len(test_loader.dataset)
        self.bma_acc = 100.0 * bma_correct / len(test_loader.dataset)
        self.best_cur_acc = max(self.best_cur_acc, self.cur_acc)
        self.best_bma_acc = max(self.best_bma_acc, self.bma_acc)
        self.best_nll = min(self.best_nll, self.nll)
        self.best_bma_nll = min(self.best_bma_nll, self.bma_nll) if bma == True else float('inf')
