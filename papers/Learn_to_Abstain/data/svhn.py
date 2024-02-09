from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple

import torch.utils.data as data
import numpy as np
import copy

from utils.utils import download_url, check_integrity, verify_str_arg


class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            clean_ratio: float = 1.0, 
            data_num_ratio: float = 1.0,
            noise_ratio: float = 1.0
    ) -> None:

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)

        self.data = np.transpose(self.data, (3, 2, 0, 1))

        self.num_class = len(np.unique(self.targets))
        assert self.num_class == 10, "Incorrect number of categories."

        # split the dataset into two parts: noisy and clean
        # we only use data_num_datio data; also, make the clean and noisy data
        data_num_each_class = int(data_num_ratio * len(self.data)/self.num_class)
        inform_data_num_each_class = int(clean_ratio * data_num_each_class)
        uninform_data_num_each_class = int(noise_ratio * data_num_each_class)

        class_remap = {0: 0, 6: 0, 1: 1, 7: 1, 2: 2, 4: 2, 3: 3, 8: 3, 5: 4, 9: 4}
        noisy_class_name = [0, 1, 2, 3, 5]

        inform_data = []
        inform_labels = []
        uninform_data = []
        uninform_labels = []

        for cls_name in range(self.num_class):
            cls_target = class_remap[cls_name]

            idx = np.where(self.targets == cls_name)[0]
            idx = idx[:data_num_each_class]

            if cls_name in noisy_class_name:
                noise_idx = idx[:uninform_data_num_each_class]
                uninform_data.append(self.data[noise_idx])
                uninform_labels.append(np.ones(len(noise_idx)) * cls_target)
            else:
                clean_idx = idx[:inform_data_num_each_class]
                inform_data.append(self.data[clean_idx])
                inform_labels.append(np.ones(len(clean_idx)) * cls_target)

        self.uninform_data = np.concatenate(uninform_data, axis=0)
        self.uninform_labels = np.concatenate(uninform_labels).astype(np.int32)
        self.inform_data = np.concatenate(inform_data, axis=0)
        self.inform_labels = np.concatenate(inform_labels).astype(np.int32)
        assert len(np.unique(self.uninform_labels)) == 5, "Incorrect number of noisy classes."

        self.uninform_datasize = len(self.uninform_labels)
        self.inform_datasize = len(self.inform_labels)
        print("Clean data size: {}\nNoisy data size: {}".format(self.inform_datasize, self.uninform_datasize))

        self.data = np.concatenate([self.uninform_data, self.inform_data], axis=0)
        self.targets = np.concatenate([self.uninform_labels, self.inform_labels], axis=0)
        print("Total number of data: {}".format(len(self.targets)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def get_uninform_labels(self):
        return copy.deepcopy(self.uninform_labels)

    def corrupte_uninform_labels(self, noisy_labels):
        self.uninform_labels[:] = noisy_labels[:]
        self.targets[:self.uninform_datasize] = noisy_labels[:]

    def get_inform_labels(self):
        return copy.deepcopy(self.inform_labels)

    def corrupte_inform_labels(self, clean_labels):
        self.inform_labels[:] = clean_labels[:]
        self.targets[self.uninform_datasize:] = clean_labels[:]

    # def update_all_labels(self, new_labels):
    #     self.uninform_labels = new_labels[:self.uninform_datasize]
    #     self.inform_labels = new_labels[self.uninform_datasize:]
    #     self.targets[:] = new_labels[:]

    # def update_partial_labels(self, idx, new_labels):

    #     assert isinstance(new_labels, type(self.targets))
    #     assert new_labels.shape == self.targets[idx].shape

    #     self.targets[idx] = new_labels
    #     self.uninform_labels = self.targets[:self.uninform_datasize]
    #     self.inform_labels = self.targets[self.uninform_datasize:]

    # def select_data(self, idx):
    #     self.data = self.data[idx]
    #     self.targets = self.targets[idx]
    #     self.uninform_data = self.uninform_data[[x for x in idx if x < self.uninform_datasize]]
    #     self.inform_data = self.inform_data[[x-self.uninform_datasize for x in idx if x >= self.uninform_datasize]]
    #     self.uninform_labels = self.uninform_labels[[x for x in idx if x < self.uninform_datasize]]
    #     self.inform_labels = self.inform_labels[[x-self.uninform_datasize for x in idx if x >= self.uninform_datasize]]
    #     self.uninform_data_size = len(self.uninform_labels)
    #     self.inform_data_size = len(self.inform_labels)

    # def get_targets(self):
    #     return torch.cat([self.uninform_labels, self.inform_labels])


if __name__ == '__main__':
    d = SVHN('data')
    print(d[0][0].size)
