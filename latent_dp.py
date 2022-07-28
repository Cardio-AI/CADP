from typing import Union
from typing import Tuple
from typing import List
from typing import Optional
from pathlib import Path
import os
import json

import torchvision.datasets
import tqdm
import torch
from torch.distributions import Laplace
from torch.distributions import Normal
from torch.distributions import Distribution
from torchvision import transforms
import FrEIA.framework as ff
import data
import models
import utils

_tqdm = tqdm.notebook.tqdm if utils.in_notebook() else tqdm.tqdm

# l1_sensitivity = lambda z: torch.abs(z).max(dim=0)[0]
# l2_sensitivity = lambda z: torch.sqrt(torch.sum(torch.square(z), dim=0) + 1e-6)

def norm_clip(z: torch.Tensor, s: float, ord: int = 1) -> torch.Tensor:
    norm = torch.linalg.norm(z, ord=ord)
    if norm > s:
        return s * (z / norm)
    else:
        return z

class DPMechanism:
    def __init__(self,
                 d: Distribution,
                 scale: float):
        # self.ds = [d(0, scale) for scale in scales]
        self.d = d(0, scale)

    def norm_clip(self, z: Union[torch.Tensor, Tuple[torch.Tensor,...]]) -> Union[torch.Tensor, Tuple[torch.Tensor,...]]:
        raise NotImplementedError

    def __call__(self,
                 z: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        z = self.norm_clip(z)
        if type(z) == tuple:
            return tuple([_z + self.d.sample(_z.size()).cuda() for _z in z])
        else:
            return z + self.d.sample(z.size()).cuda()

class GaussianMechanism(DPMechanism):
    def __init__(self,
                 epsilon: Union[float, torch.Tensor],
                 delta: Union[float, torch.Tensor],
                 sensitivity: float = 1.):
                 # zs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        # if type(zs) != tuple:
        #     zs = [zs]
        # sensitivity = [l2_sensitivity(z) for z in zs]
        self.sensitivity = sensitivity
        scale = torch.sqrt(2 * torch.log(1.25 / delta)) * sensitivity / epsilon
        super().__init__(d=Normal, scale=scale)

    def norm_clip(self, z: Union[torch.Tensor, Tuple[torch.Tensor,...]]) -> Union[torch.Tensor, Tuple[torch.Tensor,...]]:
        if type(z) == tuple:
            z = tuple([norm_clip(_z, s, 2) for _z, s in zip(z, self.sensitivity)])
        else:
            z = norm_clip(z, self.sensitivity, 2)
        return z

class LaplaceMechanism(DPMechanism):
    def __init__(self,
                 epsilon: Union[float, torch.Tensor],
                 sensitivity: float = 1.):
                 # zs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        # if type(zs) != tuple:
        #     zs = [zs]
        # sensitivity = [l1_sensitivity(z) for z in zs]
        self.sensitivity = sensitivity
        # if type(sensitivity) != list:
        #  sensitivity = [sensitivity]
        # scales = [s / epsilon for s in sensitivity]
        super().__init__(d=Laplace, scale=sensitivity / epsilon)

    def norm_clip(self, z: Union[torch.Tensor, Tuple[torch.Tensor,...]]) -> Union[torch.Tensor, Tuple[torch.Tensor,...]]:
        if type(z) == tuple:
            z = tuple([norm_clip(_z, self.sensitivity, 1) for _z in z])
        else:
            z = norm_clip(z, self.sensitivity, 1)
        return z

# def init_dp_mechanism(inn: ff.ReversibleGraphNet,
#                       data_loader: torch.utils.data.DataLoader,
#                       epsilon: float,
#                       # cond_sizes: List[Tuple[int,...]],
#                       n_classes: int,
#                       delta: Optional[torch.tensor] = torch.tensor(1),
#                       mechanism: str = "laplace") -> DPMechanism:
#
#     with torch.no_grad():
#         zs = torch.cat([inn(x.cuda(), data.make_cond(y.cuda(), inn.cond_sizes, data_loader.batch_size, n_classes))[0].cpu() for x,y in _tqdm(data_loader)], dim=0)
#     if mechanism == "gaussian":
#         dp_mechanism = GaussianMechanism(epsilon, zs, delta=delta)
#     else:
#         dp_mechanism = LaplaceMechanism(epsilon, zs)
#     return dp_mechanism

def apply_dp(inn: ff.ReversibleGraphNet,
             x: torch.Tensor,
             y: torch.Tensor,
             n_classes: int,
             dp_mechanism: DPMechanism,
             multiclass: bool = False) -> torch.Tensor:
    with torch.no_grad():
        cond = data.make_cond(y.cuda(), inn.cond_sizes, x.size(0), n_classes, multiclass)
        z, _ = inn(x, cond)
        z_dp = dp_mechanism(z)
        rev, _ = inn(z_dp, cond, rev=True)
    return rev

def privatize(inn: ff.ReversibleGraphNet,
              dataset: torch.utils.data.Dataset,
              n_classes: int,
              dp_mechanism: DPMechanism,
              multiclass: bool,
              save_path: Optional[str],
              batch_size: int = 16,
              class_names: Optional[List[str]] = None) -> torch.utils.data.Dataset:
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if class_names is not None:
        save_paths = []
        for cn in class_names:
            sp = Path.joinpath(save_path, cn)
            sp.mkdir(parents=True, exist_ok=True)
            save_paths.append(sp)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    for i, (x, y) in enumerate(_tqdm(data_loader)):
        x, y = x.cuda(), y.cuda()
        x_dp = apply_dp(inn, x, y, n_classes, dp_mechanism, multiclass)
        x_dp = torch.clamp(x_dp, 0, 1)
        for j, (img, label) in enumerate(zip(x_dp.cpu(), y)):
            img = transforms.ToPILImage()(img)
            if class_names is not None:
                save_dir = str(save_paths[label.item()])
            else:
                save_dir = str(save_path)
            img.save(f"{save_dir}/{len(os.listdir(save_dir)) + 1}.png")
    priv_ds = torchvision.datasets.ImageFolder(root=str(save_path), transform=dataset.transform)
    return priv_ds


def privatize_pipeline(exp_name: str,
                       save_path: str,
                       data_path: str,
                       epsilon: float,
                       sensitivity: float = 1.,
                       mechanism: str = "laplace",
                       dataset: str = "train", # "test" "train"
                       delta: torch.Tensor = torch.tensor(1.)) -> torch.utils.data.Dataset:
    # timestamp = ""
    # exp_name = f"mnist_conv_cond/{timestamp}"
    with open(f"./experiments/{exp_name}/summary.json", 'r') as f:
        config = json.load(f)

    img_dims = getattr(data, f"{config['dataset']}_img_dims")
    n_classes = getattr(data, f"{config['dataset']}_n_classes")
    class_names = getattr(data, f"{config['dataset']}_class_names")
    conditional = config["conditional"]
    batch_size = config["batch_size"]

    inn = getattr(models, f"{config['model_name']}_model")(n_blocks_fc=config["n_blocks_fc"],
                                                           img_dims=img_dims,
                                                           n_classes=n_classes,
                                                           ica=config["ica"],
                                                           internal_width=config["internal_width"],
                                                           clamp=config["clamp"],
                                                           init_scale=config["init_scale"],
                                                           depths=config["depths"],
                                                           channels=config["channels"],
                                                           conditional=conditional, )
    checkpoint = torch.load(f"./experiments/{exp_name}/checkpoint.pt")
    inn.load_state_dict(checkpoint["model"])

    dataset = getattr(data, f"{config['dataset']}_{dataset}_data")(f"{data_path}/{config['dataset']}")

    # dp_mechanism = init_dp_mechanism(inn=inn,
    #                                  data_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=True),
    #                                  epsilon=epsilon,
    #                                  n_classes=n_classes,
    #                                  mechanism=mechanism,
    #                                  delta=delta)

    if mechanism == "gaussian":
        dp_mechanism = GaussianMechanism(epsilon=epsilon, sensitivity=sensitivity, delta=delta)
    else:
        dp_mechanism = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)

    priv_ds = privatize(inn=inn,
                        dataset=dataset,
                        n_classes=n_classes,
                        dp_mechanism=dp_mechanism,
                        multiclass=False if config["dataset"] != "celeba" else True,
                        save_path=save_path,
                        batch_size=batch_size,
                        class_names=class_names)
    return priv_ds