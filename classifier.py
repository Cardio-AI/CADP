from typing import Tuple
from typing import Optional
from typing import Union
from typing import List
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import FrEIA.framework as ff
import data
import models
import latent_dp

class Classifier(nn.Sequential):
    def __init__(self,
                 img_dims: Tuple[int, ...],
                 n_classes: int,
                 depth: int,
                 dropout: float = 0.5,
                 dp_sgd: bool = False):
        _out_conv_size = img_dims # list(img_dims)[-2:]
        kernel_size = 3
        pad = 1
        stride = 2
        compute_out_conv_size = lambda dim: int(((_out_conv_size[dim] - kernel_size + pad) / stride) + 1)
        channels = [img_dims[0], 8, 16, 32, 64, 128, 256]
        modules = []
        for i in range(depth):
            c_in, c_out = channels[i], channels[i+1]
            modules.append(nn.Conv2d(c_in, c_out, kernel_size, padding=pad, stride=stride))
            if dp_sgd:
                modules.append(nn.GroupNorm(1, c_out))
            else:
                modules.append(nn.BatchNorm2d(c_out))
            modules.append(nn.Dropout2d(p=dropout))
            modules.append(nn.ReLU())
            _out_conv_size = (
                c_out, *[compute_out_conv_size(dim) for dim in range(1, 3)]
            )

        modules += [
            nn.Flatten(),
            nn.Linear(np.prod(_out_conv_size), 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        ]
        super().__init__(*modules)

class ClassifierPixelDP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

def train(classifier: nn.Module,
          # inn: ff.ReversibleGraphNet,
          # dp_mechanism: Optional[latent_dp.DPMechanism],
          optim: torch.optim.Optimizer,
          data_loader: torch.utils.data.DataLoader,
          n_epochs: int,
          # n_classes: int,
          save_path: str,
          dp_sgd: bool = False,
          epsilon: float = 1.,
          delta: float = 1e-2):
    import pdb;pdb.set_trace()
    if dp_sgd:
        # classifier = ModuleValidator.fix(classifier)
        privacy_engine = PrivacyEngine()
        classifier, optim, data_loader = privacy_engine.make_private_with_epsilon(
            module=classifier,
            optimizer=optim,
            data_loader=data_loader,
            # noise_multiplier=1.1,
            max_grad_norm=10.0,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=n_epochs
        )
        classifier = classifier.cuda()
    losses, accs = [], []
    for e in range(n_epochs):
        # if isinstance(data_loader.sampler, torch.utils.data.WeightedRandomSampler):
        #     idx = torch.randperm(len(data_loader.dataset))
        #     data_loader.dataset.samples = [data_loader.dataset.samples[i] for i in idx]
        #     data_loader.dataset.targets = [data_loader.dataset.targets[i] for i in idx]
        #     data_loader.dataset.imgs = [data_loader.dataset.imgs[i] for i in idx]

        e_losses, e_accs = [], []
        for i, (x, y) in enumerate(tqdm(data_loader, leave=False)):
            optim.zero_grad()
            x, y = x.cuda(), y.cuda()
            # if dp_mechanism is not None:
            #     x_dp = latent_dp.apply_dp(inn, x, y, n_classes, dp_mechanism)
            #     pred = classifier(x_dp)
            # else:

            pred = classifier(x)
            xent = torch.nn.functional.cross_entropy(pred, y)
            xent.backward()
            optim.step()
            e_losses.append(xent.item())

            acc = (torch.argmax(pred, dim=1) == y).sum() / y.size(0)
            e_accs.append(acc.item())

            if dp_sgd:
                epsilon_spent = privacy_engine.get_epsilon(delta)
                #if epsilon_spent > epsilon:
                #    break
            else:
                epsilon_spent = epsilon

        e_loss = np.mean(e_losses)
        e_acc = np.mean(e_accs)
        losses.append(e_loss)
        accs.append(e_acc)
        torch.save(classifier.state_dict(), f"{save_path}/checkpoint.pt")
        print(f"E: {e} | L: {e_loss:.4f} | A: {e_acc * 100:.4f} | eps: {epsilon_spent:.5f}")

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(range(len(losses)), losses)
        axs[0].set_xlabel("iteration")
        axs[0].set_ylabel("loss")
        axs[1].plot(range(len(accs)), accs)
        axs[1].set_xlabel("iteration")
        axs[1].set_ylabel("acc")
        fig.tight_layout()
        fig.savefig(f"{save_path}/train_stats.png")
        plt.close()

if __name__ == "__main__":
    import argparse
    import json
    import time
    import random
    import importlib
    from pathlib import Path

    timestamp = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-e', type=str) # exp_name/timestamp
    parser.add_argument('--data_path', '-d', type=str) # data_path/priv_ds/epsilon
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--epsilon', type=float, default=0.)
    # parser.add_argument('--delta', type=float, default=1e-2)
    parser.add_argument('--dp_sgd', action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = importlib.import_module(f"configs.classifier.{args.config}")

    # with open(f"./experiments/{args.exp_name}/summary.json", 'r') as f:
    #     inn_config = json.load(f)

    img_dims = getattr(data, f"{config.dataset}_img_dims")
    n_classes = getattr(data, f"{config.dataset}_n_classes")
    # conditional = inn_config["conditional"]
    if args.dp_sgd:
        exp_path = Path(f"./experiments/{args.exp_name}/classifier/dp_sgd/{args.epsilon}/{timestamp}")
    elif args.epsilon == 0.:
        exp_path = Path(f"./experiments/{args.exp_name}/classifier/no_dp/{timestamp}")
    else:
        exp_path = Path(f"./experiments/{args.exp_name}/classifier/{args.epsilon}/{timestamp}")
    exp_path.mkdir(exist_ok=True, parents=True)

    summary = {k: getattr(config, k) for k in ["depth", "dropout", "n_epochs", "lr", "batch_size", "weight_decay", "betas", "dataset"]}
    summary["epsilon"] = args.epsilon
    with open(f"{exp_path}/summary.json", 'w') as f:
        json.dump(summary, f, indent=4, sort_keys=True)

    # inn = getattr(models, f"{inn_config['model_name']}_model")(n_blocks_fc=inn_config["n_blocks_fc"],
    #                                                            img_dims=img_dims,
    #                                                            n_classes=n_classes,
    #                                                            ica=inn_config["ica"],
    #                                                            internal_width=inn_config["internal_width"],
    #                                                            clamp=inn_config["clamp"],
    #                                                            init_scale=inn_config["init_scale"],
    #                                                            depths=inn_config["depths"],
    #                                                            channels=inn_config["channels"],
    #                                                            conditional=conditional,)
    #
    # checkpoint = torch.load(f"./experiments/{args.exp_name}/checkpoint.pt")
    # inn.load_state_dict(checkpoint["model"])
    # inn.eval()

    classifier = Classifier(depth=config.depth,
                            n_classes=n_classes,
                            dropout=config.dropout,
                            img_dims=img_dims,
                            dp_sgd=args.dp_sgd).cuda()
    optim = torch.optim.Adam(classifier.parameters(),
                             lr=config.lr,
                             betas=config.betas,
                             weight_decay=config.weight_decay)

    if args.dp_sgd or args.epsilon == 0.:
        dataset = getattr(data, f"{config.dataset}_train_data")(f"{args.data_path}/{config.dataset}")
    else:
        t = getattr(data, f"{args.config}_transform", None)
        dataset = torchvision.datasets.ImageFolder(f"{args.data_path}/{args.epsilon}", transform=t)
    t = getattr(data, f"{args.config}_transform_classifier", None)
    dataset.transform = t(dataset)

    if not args.dp_sgd:
        _, counts = torch.unique(torch.tensor(dataset.targets), return_counts=True)
        weights = 1 - counts / counts.sum()
        sampler = torch.utils.data.WeightedRandomSampler(weights[dataset.targets], len(dataset))
        # shuffle = False
    else:
        sampler = None
    delta = 1 / len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              sampler=sampler,
                                              batch_size=config.batch_size,
                                              num_workers=4,
                                              pin_memory=True)#,
                                              # drop_last=True,
                                              # shuffle=shuffle)
    # labels = torch.cat([y for x, y in data_loader], dim=0)
    # print(torch.unique(labels, return_counts=True))

    # if config.dp_mechanism is not None:
    #     dp_mechanism = latent_dp.init_dp_mechanism(inn=inn,
    #                                                data_loader=data_loader,
    #                                                n_classes=n_classes,
    #                                                mechanism=config.dp_mechanism,
    #                                                epsilon=config.epsilon,
    #                                                delta=torch.tensor(config.delta))
    # else:
    #     dp_mechanism = None

    train(classifier=classifier,
          # inn=inn,
          # dp_mechanism=dp_mechanism,
          optim=optim,
          data_loader=data_loader,
          n_epochs=config.n_epochs,
          # n_classes=n_classes,
          save_path=str(exp_path),
          dp_sgd=args.dp_sgd,
          epsilon=args.epsilon,
          delta=delta)