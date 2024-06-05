import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as VF
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg
import numpy as np
from PIL import Image
import h5py

ARCHIVE_META = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}

META_FILE = "meta.bin"


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: str = "train", partial=True, **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super().__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        if partial:    
            choose_ind = torch.where(torch.tensor(self.targets) < 10)[0]
        else:
            choose_ind = range(len(self.targets))
        self.imgs = [self.imgs[i] for i in choose_ind]
        self.data = [self.samples[i] for i in choose_ind]
        self.targets = [self.targets[i] for i in choose_ind]
        self.labels_c, self.labels_t = self.targets, self.targets
    
        self.clean_num = len(self.imgs)

        self.troj_data = []
        self.troj_labels_c = np.array([])
        self.troj_labels_t = np.array([])
        
        self.use_transform = True

    
    def __len__(self):
        return len(self.data)+len(self.troj_data)
        
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        if index < self.clean_num:
            pth, labels_c, labels_t = self.samples[index], self.labels_c[index], self.labels_t[index]
            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = self.loader(pth[0])
        else:
            img, labels_c, labels_t = self.troj_data[index-self.clean_num], self.troj_labels_c[index-self.clean_num], self.troj_labels_t[index-self.clean_num]
        
        if self.use_transform and self.transform is not None:
            img = self.transform(img)
        else:
            img = VF.resize(VF.to_tensor(img), (224, 224))
            
        if self.target_transform is not None:
            labels_c = self.target_transform(labels_c)
            labels_t = self.target_transform(labels_t)
        else:
            labels_c = torch.tensor(labels_c)
            labels_t = torch.tensor(labels_t)

        return index, img, labels_c, labels_t
    
    
    def insert_data(self, new_data: List, new_labels_c: np.ndarray, new_labels_t: np.ndarray) -> None:
        assert isinstance(new_data, List), "data need to be a list, but find " + str(type(new_data)) 
        assert isinstance(new_labels_c, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_c))
        assert isinstance(new_labels_t, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_t))
        self.troj_data += new_data
        self.troj_labels_c = np.append(self.troj_labels_c, new_labels_c).astype(np.int64)
        self.troj_labels_t = np.append(self.troj_labels_t, new_labels_t).astype(np.int64)
    
        
    def select_data(self, indices: np.ndarray) -> None:
        assert isinstance(indices, np.ndarray), "indices need to be np.ndarray, but find " + str(type(indices))
        self.data = [self.data[i] for i in indices]
        self.labels_c = np.array(self.labels_c)[indices]
        self.labels_t = np.array(self.labels_t)[indices]
        self.clean_num = len(self.labels_c)
        
    
    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == "train":
                parse_train_archive(self.root)
            elif self.split == "val":
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    # my adds-on method
    def save_selected_images(self, save_dir: str, **kwargs) -> None:
        
        hf = h5py.File(os.path.join(save_dir, f'imagenet-10class-{self.split}'), 'w')
        
        for index in range(len(self.data)):
            
            group_index = hf.create_group(f'{index}')
            
            pth, labels_c, labels_t = self.data[index], self.labels_c[index], self.labels_t[index]
            img = np.array(self.loader(pth[0]))
            
            group_index.create_dataset(f'data_{index}', data=img)
            group_index.create_dataset(f'labels_c_{index}', data=labels_c)
            group_index.create_dataset(f'labels_t_{index}', data=labels_t)
            
            print(f"Progress: {index/len(self.data)*100:.3f}%", end='\r')

        hf.close()

def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = (
            "The meta file {} is not present in the root directory or is corrupted. "
            "This file is automatically created by the ImageNet dataset."
        )
        raise RuntimeError(msg.format(file, root))


def _verify_archive(root: str, file: str, md5: str) -> None:
    if not check_integrity(os.path.join(root, file), md5):
        msg = (
            "The archive {} is not present in the root directory or is corrupted. "
            "You need to download it externally and place it in {}."
        )
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root: str, file: Optional[str] = None) -> None:
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(", ")) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
        with open(file) as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_train_archive(root: str, file: Optional[str] = None, folder: str = "train") -> None:
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the train images archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_img_train.tar'
        folder (str, optional): Optional name for train images folder. Defaults to
            'train'
    """
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    train_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), train_root)

    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    for archive in archives:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def parse_val_archive(
    root: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val"
) -> None:
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            'val'
    """
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]

    _verify_archive(root, file, md5)

    val_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), val_root)

    images = sorted(os.path.join(val_root, image) for image in os.listdir(val_root))

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))
        

class ImagenetDownSample(Dataset):
    
    def __init__(self, 
                 root:str, 
                 split:str='train', 
                 transform:transforms = None,
                 target_transform: transforms = None, 
                 **kwargs):
        
        self.split = split
        self.root  = root
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = int(kwargs['config']['dataset']['imagenet']['IMG_SIZE'])
        
        self.hf = h5py.File(os.path.join(root, f'imagenet-10class-{self.split}'), 'r')
        self.labels_c = [int(np.array(self.hf.get(f'{index}/labels_c_{index}'))) for index in range(len(self.hf))]
        self.labels_t = [int(np.array(self.hf.get(f'{index}/labels_t_{index}'))) for index in range(len(self.hf))]
        
        self.labels_c = np.array(self.labels_c)
        self.labels_t = np.array(self.labels_t)
        self.indices  = list(range(len(self.hf)))
        self.active_indices = list(range(len(self.hf)))
        
        self.clean_num = len(self.active_indices)

        self.troj_data = []
        self.troj_labels_c = np.array([])
        self.troj_labels_t = np.array([])
        
        self.use_transform = True
    
    def __len__(self) -> int:
        return len(self.active_indices)+len(self.troj_data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        if index < self.clean_num:
            activeindex = self.active_indices[index]
            # img, labels_c, labels_t = self.data[index], self.labels_c[index], self.labels_t[index]
            img = Image.fromarray(np.array(self.hf.get(f'{activeindex}/data_{activeindex}')))
            labels_c, labels_t = self.labels_c[activeindex], self.labels_t[activeindex]
        else:
            img, labels_c, labels_t = self.troj_data[index-self.clean_num], self.troj_labels_c[index-self.clean_num], self.troj_labels_t[index-self.clean_num]
        
        if self.use_transform and self.transform is not None:
            img = self.transform(img)
        else:
            img = VF.resize(VF.to_tensor(img), (self.img_size, self.img_size))
            
        if self.target_transform is not None:
            labels_c = self.target_transform(labels_c)
            labels_t = self.target_transform(labels_t)
        else:
            labels_c = torch.tensor(labels_c)
            labels_t = torch.tensor(labels_t)

        index = activeindex if index < self.clean_num else index
        return index, img, labels_c, labels_t
    

    def insert_data(self, new_data: List, new_labels_c: np.ndarray, new_labels_t: np.ndarray) -> None:
        assert isinstance(new_data, List), "data need to be a list, but find " + str(type(new_data)) 
        assert isinstance(new_labels_c, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_c))
        assert isinstance(new_labels_t, np.ndarray), f"labels need to be a np.ndarray, but find " + str(type(new_labels_t))
        self.troj_data += new_data
        self.troj_labels_c = np.append(self.troj_labels_c, new_labels_c).astype(np.int64)
        self.troj_labels_t = np.append(self.troj_labels_t, new_labels_t).astype(np.int64)
    
        
    def select_data(self, indices: np.ndarray) -> None:
        assert isinstance(indices, np.ndarray), "indices need to be np.ndarray, but find " + str(type(indices))
        self.active_indices = [self.indices[i] for i in indices]
        self.clean_num = len(self.active_indices)

if __name__  == '__main__':

    # create downsampled dataset 
    trainset = ImageNet(root="/Path/To/Your ImageNet Folder", split='train', partial=True)
    testset  = ImageNet(root="/Path/To/Your ImageNet Folder", split='val', partial=True)
    
    trainset.save_selected_images(save_dir="./data")
    testset.save_selected_images(save_dir="./data")

    # load downsampled images
    trainset = ImagenetDownSample(root='./data', split='train')
    print(trainset)