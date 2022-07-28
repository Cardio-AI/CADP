import pdb
from typing import Dict, Any, Tuple, Union, List, Callable
import functools
import os
from PIL import Image
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel
import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import monai
import torchvision.datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import utils

def make_cond_conv(labels: torch.Tensor,
                   cond_size: Tuple[int,...],
                   batch_size: int,
                   n_classes: int) -> torch.Tensor:
    cond = torch.einsum('ij,ijkl->ijkl',
                        torch.scatter(torch.zeros((batch_size, n_classes)).cuda(), 1, labels.view(-1, 1), 1.),
                        torch.ones((batch_size, *cond_size)).cuda())
    return cond

def make_cond_fc(labels: torch.Tensor,
                 batch_size: int,
                 n_classes: int) -> torch.Tensor:
    return torch.scatter(torch.zeros((batch_size, n_classes)).cuda(), 1, labels.view(-1,1), 1.)

def make_cond(labels: torch.Tensor,
              cond_sizes: List[Tuple[int,...]],
              batch_size: int,
              n_classes: int,
              multiclass: bool = False) -> List[torch.Tensor]:
    if multiclass:
        cond = [(labels[...,None,None] if len(cs) == 3 else labels) * torch.ones((batch_size, *cs)).to(labels.device) for cs in cond_sizes]
    else:
        cond = [make_cond_conv(labels, cs, batch_size, n_classes)
                if len(cs) == 3 else make_cond_fc(labels, batch_size, n_classes)
                for cs in cond_sizes]
    return cond

# def make_class_cond(y: torch.Tensor, n_classes: int) -> torch.Tensor:
#     return torch.zeros((y.size(0), n_classes), device="cuda").scatter_(1, y.view(-1, 1), 1.)

def unnormalize(x: torch.Tensor, data_std: float = 1., data_mean: float = 0.) -> torch.Tensor:
    return x * data_std + data_mean

def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     n_samples: int = 10000) -> Tuple[float, ...]:
    data_iter = iter(dataset)
    xs = torch.cat([next(data_iter)[0][None] for _ in utils.tqdm_()(range(min(n_samples, len(dataset))))], dim=0)
    mean = xs.mean(dim=[0,2,3])
    std = xs.std(dim=[0,2,3])
    # only works for grayscale for now
    return mean.item(), std.item()

to_gray = T.Grayscale(num_output_channels=1)
def alb_transform(image: np.ndarray,
                  transforms: A.Compose,
                  grayscale: bool = True) -> torch.Tensor:
    if grayscale:
        image = to_gray(image)
    image = np.array(image).astype(np.float32)
    # if np.max(image) > 1:
    #     image /= 255
    return transforms(image=image)["image"].float()

mnist_transform = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor(),])
mnist_train_data = functools.partial(torchvision.datasets.MNIST, train=True, transform=T.ToTensor(), download=True)
mnist_test_data = functools.partial(torchvision.datasets.MNIST, train=False, transform=T.ToTensor(), download=True)
mnist_unnormalize = functools.partial(unnormalize, data_std=1., data_mean=0.)
mnist_img_dims = (1, 28, 28)
mnist_n_classes = 10
mnist_class_names = [str(i) for i in range(10)]
# mnist_make_class_cond = lambda x, y: torch.zeros((y.size(0), 10), device="cuda").scatter_(1, y.view(-1,1), 1.)

mnist_transform_classifier = lambda dataset: mnist_transform

class GetImageAndLabel:
    def __call__(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        return data["image"], data["label"]

class Normalize:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # try:
        data["image"] -= data["image"].min()
        # data["image"][data["image"] == np.nan] = 0
        data["image"] /= data["image"].max() # - data["image"].min())
        data["image"][np.isnan(data["image"])] = 0.
        # except:
        #     import pdb;pdb.set_trace()
        # data["image"] /= 255
        return data

mednist_transform = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.AddChanneld(keys=["image"]),
        Normalize(),
        # ScaleIntensityd(keys="image"),
        # monai.transforms.NormalizeIntensityd(keys=["image"]),
        monai.transforms.ToTensord(keys=["image", "label"]),
        GetImageAndLabel()
])

mednist_train_data = functools.partial(monai.apps.MedNISTDataset, section="training", transform=mednist_transform, download=True)
mednist_test_data = functools.partial(monai.apps.MedNISTDataset, section="test", transform=mednist_transform, download=True)
mednist_unnormalize = functools.partial(unnormalize, data_std=1., data_mean=0.)
mednist_img_dims = (64, 64)
# mednist_make_class_cond = lambda x, y: torch.zeros((y.size(0), 6), device="cuda").scatter_(1, y.view(-1,1), 1.)

celeba_img_dims = (3, 128, 128)
celeba_transform = T.Compose([
    T.Resize((celeba_img_dims[1], celeba_img_dims[2])),
    T.ToTensor()
])
celeba_target_transform = lambda label: (torch.from_numpy(label.to_numpy()[0,1:].astype(np.float32) == 1).float())

class CelebA:
    def __init__(self, root: str, split: str = "train", transform: Any = None, target_transform: Any = None):
        # if not root.endswith("/celeba"):
        #     root += "/celeba"
        self.root = root
        self.img_root_path = f"{root}/img_align_celeba/img_align_celeba"

        self.attr_labels = pd.read_csv(f"{root}/list_attr_celeba.csv")

        split_df = pd.read_csv(f"{root}/list_eval_partition.csv")

        get_split_ids = lambda _id: split_df.loc[split_df["partition"] == _id]["image_id"].to_list()
        if split == "train":
            self.img_ids = split_df.loc[split_df["partition"] == 0]["image_id"].to_list()
        elif split == "test":
            self.img_ids = split_df.loc[split_df["partition"] == 1]["image_id"].to_list()
        elif split == "val":
            self.img_ids = split_df.loc[split_df["partition"] == 2]["image_id"].to_list()
        else:
            raise ValueError

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item: int) -> tuple:
        img_id = self.img_ids[item]
        img = Image.open(f"{self.img_root_path}/{img_id}")
        label = self.attr_labels.loc[self.attr_labels["image_id"] == img_id]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.img_ids)

celeba_train_data = functools.partial(CelebA, split="train", transform=celeba_transform, target_transform=celeba_target_transform)
celeba_test_data = functools.partial(CelebA, split="test", transform=celeba_transform, target_transform=celeba_target_transform)
celeba_unnormalize = functools.partial(unnormalize, data_std=1., data_mean=0.)
celeba_n_classes = 40
# celeba_train_data = functools.partial(torchvision.datasets.CelebA, split="train", target_type="attr", transform=celeba_transform, download=True)
# celeba_test_data = functools.partial(torchvision.datasets.CelebA, split="test", target_type="attr", transform=celeba_transform, download=True)

class ACDC3D:
    label_assignment = {"DCM": 0, "HCM": 1, "MINF": 2, "RV": 3, "NOR": 4}
    def __init__(self, root: str,
                 split: str = "train",
                 transform: Any = None,
                 seg_transform: Any = None,
                 target_transform: Any = None,
                 phase: str = "ED"):
        # if not root.endswith("/acdc"):
        #     root += "/acdc"
        self.root = root
        self.patients = os.listdir(root)

        self.transform = transform
        self.seg_transform = seg_transform
        self.target_transform = target_transform

        self.phase = phase # ED or ES

    def __getitem__(self, item: int) -> tuple:
        patient = self.patients[item]
        label = self.read_info(f"{self.root}/{patient}/Info.cfg")

        frame = label[self.phase]
        frame = '0' * (2 - len(frame)) + frame
        img = nibabel.load(f"{self.root}/{patient}/{patient}_frame{frame}.nii.gz")
        img = np.asanyarray(img.dataobj)

        seg = nibabel.load(f"{self.root}/{patient}/{patient}_frame{frame}_gt.nii.gz")
        seg = np.asanyarray(seg.dataobj)

        # _label = info["Group"]
        # label = np.zeros((5,)).astype(np.float32)
        # label[self.label_assignment[_label]] = 1.
        # label = sitk.ReadImage(f"{self.root}/{patient}/{patient}_frame01_gt.nii.gz")

        if self.transform is not None:
            img = self.transform(img)
        if self.seg_transform is not None:
            seg = self.seg_transform(seg)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, seg, patient

    def __len__(self) -> int:
        return len(self.patients)

    @staticmethod
    def read_info(path: str) -> Dict[str, Union[float, str]]:
        with open(path) as f:
            content = f.read()
        info = {s.split(': ')[0]: s.split(': ')[1] for s in content.split('\n') if len(s)}
        return info

acdc3d_transform = T.Compose([
    # lambda img: torch.from_numpy(sitk.GetArrayFromImage(img))
    # lambda img: torch.from_numpy(img),
    lambda img: (img - img.min()) / (img.max() - img.min())
])
acdc3d_seg_transform = torch.from_numpy
acdc3d_target_transform = None # lambda t: torch.from_numpy(t)
acdc3d_train_data = functools.partial(ACDC3D,
                                      transform=acdc3d_transform,
                                      seg_transform=acdc3d_seg_transform,
                                      target_transform=acdc3d_target_transform)
acdced_unnormalize = functools.partial(unnormalize, data_std=1., data_mean=0.)

class ACDC:
    label_assignment = {"DCM": 0, "HCM": 1, "MINF": 2, "RV": 3, "NOR": 4}
    def __init__(self, root: str,
                 split: str = "train",
                 transform: Any = None,
                 target_transform: Any = None,
                 phase: str = "ED"):

        # if not root.endswith("_2d"):
        #     root += "_2d"
        self.root = root
        self.patients = os.listdir(f"{root}/imgs")
        self.labels = pd.read_csv(f"{root}/info.csv").set_index("pat")

        self.transform = transform
        self.target_transform = target_transform

        # self.phase = phase # ED or ES

    def __getitem__(self, item: int) -> tuple:
        patient = self.patients[item]

        img = Image.open(f"{self.root}/imgs/{patient}")
        label = self.labels.loc[patient.split('.')[0]]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.patients)

acdc_img_dims = (1, 128, 128)
acdc_transform = T.Compose([
    T.Resize((acdc_img_dims[-2], acdc_img_dims[-1])),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # T.Normalize(), # already done when saved from 3d to 2d
    T.ToTensor()
])
acdc_n_classes = 5

def acdc_target_transform(label: pd.Series) -> torch.Tensor:
    acdc_group_assignment = {"DCM": 0, "HCM": 1, "MINF": 2, "RV": 3, "NOR": 4}
    # _label = torch.zeros((5,)).float()
    # _label[acdc_group_assignment[label["group"]]] = 1.
    # return _label
    return torch.tensor(acdc_group_assignment[label["group"]])

acdc_train_data = functools.partial(ACDC, transform=acdc_transform, target_transform=acdc_target_transform)
acdc_test_data = functools.partial(ACDC, transform=acdc_transform, target_transform=acdc_target_transform)
acdc_unnormalize = functools.partial(unnormalize, data_std=1., data_mean=0.)

chest_xray_img_dims = (1, 128, 128)
chest_xray_n_classes = 2
chest_xray_class_names = ["Healthy", "Pneumonia"]
chest_xray_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((chest_xray_img_dims[1], chest_xray_img_dims[2])),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.ToTensor()
])
chest_xray_transform_classifier_alb = lambda mean, std: A.Compose([
    A.Resize(height=128, width=128),
    A.SafeRotate(limit=5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(mean,), std=(std,)),
    ToTensorV2()
])
def chest_xray_transform_classifier(dataset: torch.utils.data.Dataset) -> Callable:
    dataset.transform = chest_xray_transform
    mean, std = get_mean_and_std(dataset)
    t = oct_transform_classifier_alb(mean, std)
    return lambda image: alb_transform(image, t)
chest_xray_train_data = lambda root: torchvision.datasets.ImageFolder(root=f"{root}/train", transform=chest_xray_transform)
chest_xray_test_data = lambda root: torchvision.datasets.ImageFolder(root=f"{root}/test", transform=chest_xray_transform)
chest_xray_unnormalize = functools.partial(unnormalize, data_std=1., data_mean=0.)

oct_img_dims = (1, 128, 128)
oct_n_classes = 4
oct_class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
oct_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((oct_img_dims[1], oct_img_dims[2])),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.ToTensor()
])
oct_transform_classifier_alb = lambda mean, std: A.Compose([
    A.Resize(height=128, width=128),
    A.SafeRotate (limit=5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(mean,), std=(std,)),
    ToTensorV2()
])
def oct_transform_classifier(dataset: torch.utils.data.Dataset) -> Callable:
    dataset.transform = oct_transform
    mean, std = get_mean_and_std(dataset)
    t = oct_transform_classifier_alb(mean, std)
    return lambda image: alb_transform(image, t)
oct_train_data = lambda root: torchvision.datasets.ImageFolder(root=f"{root}/train", transform=chest_xray_transform)
oct_test_data = lambda root: torchvision.datasets.ImageFolder(root=f"{root}/test", transform=chest_xray_transform)
oct_unnormalize = functools.partial(unnormalize, data_std=1., data_mean=0.)