import torch
import torchvision
import torch.utils.data as data
import copy
import numpy as np


class HybridMNIST(torch.utils.data.Dataset):
    """
    This dataset consists of clean Fashion-MNIST and noisy MNIST (uniform flipping)
    """

    def __init__(self, root='data',
                 train=True,
                 mnist_transform=None,
                 fashion_transform=None,
                 data_num_ratio=1.0,
                 clean_ratio=1.0,
                 noise_ratio=1.0):
        """Method to initilaize variables."""
        self.uninform_dataset = torchvision.datasets.MNIST(root, train, transform=mnist_transform, download=True)
        self.inform_dataset = torchvision.datasets.FashionMNIST(root, train, transform=fashion_transform, download=True)

        self.uninform_datasize = len(self.uninform_dataset)
        self.inform_datasize = len(self.inform_dataset)

        # index for selecting the data
        uninform_idx = np.arange(int(self.uninform_datasize * data_num_ratio * noise_ratio))
        inform_idx = np.arange(int(self.inform_datasize * data_num_ratio))

        # we only use data_num_datio data
        self.uninform_datasize = len(uninform_idx)
        self.inform_datasize = len(inform_idx)

        self.uninform_dataset.targets = self.uninform_dataset.targets[uninform_idx]
        self.uninform_dataset.data  = self.uninform_dataset.data[uninform_idx]
        self.inform_dataset.targets = self.inform_dataset.targets[inform_idx]
        self.inform_dataset.data = self.inform_dataset.data[inform_idx]

        # collect the labels of Fashion MNIST and their indices
        print("Preprocessing Fashion MNIST ...")
        label_list = []
        num_data_each_class = int(clean_ratio*self.inform_datasize//10)
        for i in range(10):
            # import ipdb; ipdb.set_trace()
            idx_arr = np.where(self.inform_dataset.targets == i)[0]
            idx_arr = idx_arr[:num_data_each_class]
            label_list.append(idx_arr)
        inform_data_idx = np.concatenate(label_list)
        self.inform_datasize = len(inform_data_idx)  # update the fashion MNIST dataset size
        self.inform_dataset.targets = self.inform_dataset.targets[inform_data_idx]  # update the target labels
        self.inform_dataset.data = self.inform_dataset.data[inform_data_idx]  # update the image data

        # store the mnist labels for future use
        self.uninform_labels = self.uninform_dataset.targets
        self.inform_labels = self.inform_dataset.targets

        self.targets = np.concatenate([self.uninform_labels, self.inform_labels])

        print("Preprocessing Done.")
        print("MNIST dataset size: {}\nFashion-MNIST dataset size: {}".format(self.uninform_datasize, self.inform_datasize))

    def __getitem__(self, index):
        if index < self.uninform_datasize:  # use mnist dataset
            image, label = self.uninform_dataset[index]
        else:
            image, label = self.inform_dataset[index - self.uninform_datasize]

        return index, image, label

    def __len__(self):
        return self.uninform_datasize + self.inform_datasize

    def get_uninform_labels(self):
        return copy.deepcopy(self.uninform_labels)

    def get_inform_labels(self):
        return copy.deepcopy(self.inform_labels)

    def corrupte_uninform_labels(self, noisy_labels):
        self.uninform_dataset.targets = noisy_labels
        self.uninform_labels = noisy_labels
        self.targets[:self.uninform_datasize] = noisy_labels

    def corrupte_inform_labels(self, noisy_labels):
        self.inform_dataset.targets = noisy_labels
        self.inform_labels = noisy_labels
        self.targets[self.uninform_datasize:] = noisy_labels

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    fashion_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    testdata = HybridMNIST(train=False, mnist_transform=mnist_transform, fashion_transform=fashion_transform)

    index = 10000
    print("MNIST labels: ", np.unique(np.array(testdata.mnist_dataset.targets)))
    print("Fashion MNIST labels: ", np.unique(np.array(testdata.fashion_dataset.targets)))

    print("MNIST image size: ", testdata[0][0].shape)
    print("Fashion MNIST image size: ", testdata[-1][0].shape)
