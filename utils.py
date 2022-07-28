from typing import Tuple, Dict, Any, Union, Optional
import json
import numpy as np
import torch
import FrEIA.framework as ff
import data

def interpolate(z1: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                z2: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                model: ff.GraphINN) -> torch.Tensor:
    if type(z1) == tuple:
        z = tuple([torch.cat([z1[j] * (i - 1) + z2[j] * i for i in torch.arange(0, 1.1, 0.1)], dim=0).cuda().float() for j in range(len(z1))])
    else:
        z = torch.cat([(z1 * (1 - i) + z2 * i)[None] for i in torch.arange(0, 1.1, 0.1)], dim=0).cuda().float()
    with torch.no_grad():
        rev, _ = model(z, rev=True)
    return rev.cpu()

def sample_outputs(size: Tuple[int, ...],
                   sigma: float = 1.,):
    return sigma * torch.cuda.FloatTensor(*size).normal_()

def get_all_zs_and_labels(model: ff.GraphINN,
                          data_loader: torch.utils.data.DataLoader,
                          conditional: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    zs, ys = [], []
    with torch.no_grad():
        for x, labels in data_loader:
            # cond = make_cond(labels)
            z, jac = model(x)  # , cond)
            zs.append(z.cpu())
            ys.append(labels)
    return torch.cat(zs, dim=0), torch.cat(labels, dim=0)

def get_mean_rec_and_z_per_label(model: ff.GraphINN,
                                 zs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                 labels: torch.Tensor,
                                 conditional: bool = False) -> Tuple[torch.Tensor, Dict[Any, torch.Tensor]]:
    unique_labels = torch.unique(labels)
    if type(zs) == tuple:
        mean_z_per_label = {
            i.data: tuple([torch.mean(z[labels == i], dim=0) for z in zs]) for i in unique_labels # range(10)
        }
        mean_proj = tuple([torch.stack([value[i] for value in mean_z_per_label.values()]).float().cuda() for i in range(len(zs))])
    else:
        mean_z_per_label = {
            # i: torch.mean(torch.from_numpy(pca.inverse_transform(z_pca[labels == i])), dim=0) for i in range(10)
            i.data: torch.mean(zs[labels == i], dim=0) for i in unique_labels
        }
        mean_proj = torch.stack(list(mean_z_per_label.values())).float().cuda()

    if conditional:
        if len(labels.shape) == 1:
            cond = data.make_class_cond(unique_labels, len(unique_labels))
        else:
            cond = labels
    else:
        cond = None

    with torch.no_grad():
        try:
            rev, _ = model(mean_proj, cond, rev=True)
        except RuntimeError:
            rev = torch.cat([model(mean_proj[i:i+1], cond[i:i+1], rev=True)[0] for i in range(len(unique_labels))], dim=0)
    return rev, mean_z_per_label

def exp_summary(save_path: str, **kwargs) -> None:
    del kwargs["config"], kwargs["exp_path"]
    with open(f"{save_path}/summary.json", "w") as f: #f"./experiments/{kwargs['exp_name']}/summary.json", 'w') as f:
        json.dump(kwargs, f, indent=4, sort_keys=True)
    exp_string = '\n'.join([f"{ name + ' ' * (25 - len(name))}=\t\t{value}" for name, value in kwargs.items()])
    exp_string = "\n\n" + exp_string + "\n\n"
    print(exp_string)

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def tqdm_() -> Any:
    if in_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm