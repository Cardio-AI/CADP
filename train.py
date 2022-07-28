import sys
from typing import Optional, Tuple, List, Any
from pathlib import Path
import time
import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import torch
import FrEIA.framework as ff
from utils import sample_outputs
import models
import data
import viz
from utils import exp_summary
from dim_reduction import LatentPCA

class dummy_loss(object):
    def item(self):
        return 1.

def train_step(model: ff.InvertibleModule,
               optim: torch.optim.Optimizer,
               x: torch.Tensor,
               y: torch.Tensor,
               n_classes: int,
               ica: bool = True,
               do_fwd: bool = True,
               do_rev: bool = False,
               input_noise_sigma: float = 0.1,
               latent_noise_sigma: float = 1e-2,
               empirical_var: bool = True,
               # scale_l_fwd: bool = False,
               cond: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:

    optim.zero_grad()

    x += input_noise_sigma * torch.cuda.FloatTensor(x.shape).normal_()

    if do_fwd:
        z, jac = model(x, cond)

        if ica:
            if empirical_var:
                sig = torch.stack([z[y == i].std(0, unbiased=False) for i in range(n_classes)])
                # negative log-likelihood for gaussian in latent space
                nll = 0.5 + torch.log(sig[y]).mean(1) + 0.5 * np.log(2 * np.pi)
                nll -= jac / torch.prod(torch.tensor(x.size()[1:]))
                l_fwd = nll.mean()
            else:
                m = model.mu[y]
                ls = model.log_sig[y]

                # negative log-likelihood for gaussian in latent space
                nll = torch.mean(0.5 * (z - m) ** 2 * torch.exp(-2 * ls) + ls, 1) + 0.5 * np.log(2 * np.pi)
                l_fwd = torch.mean(nll - jac / torch.prod(torch.tensor(x.size()[1:])))
            # print(z.mean().item(), z.max().item(), z.min().item())
        else:
            if type(z) == tuple:
                zz = sum(torch.sum(o ** 2, dim=1) for o in z)
            else:
                zz = torch.sum(z ** 2, dim=1)

            nll = 0.5 * zz - jac

            l_fwd = torch.mean(nll)
            l_fwd /= torch.prod(torch.tensor(x.size()[1:])) # 2 * x.size(-2) * x.size(-1)

        l_fwd.backward(retain_graph=True)#do_rev)
    else:
        with torch.no_grad():
            z, jac = model(x, cond)
        l_fwd = dummy_loss()

    if do_rev:
        if type(z) == tuple:
            samples_noisy = [sample_outputs(o.size(), latent_noise_sigma) + o.data for o in output]
        else:
            samples_noisy = sample_outputs(z.size(), latent_noise_sigma) + z.data
        x_rec, _ = model(samples_noisy, cond, rev=True)

        l_rev = torch.mean((x - x_rec) ** 2)
        l_rev.backward()
    else:
        l_rev = dummy_loss()
        x_rec = torch.randn(x.size())

    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

    optim.step()

    if type(z) == tuple:
        z = [o.detach() for o in z]
    else:
        z = z.detach()

    return [l_fwd, l_rev], x_rec.detach(), z

def train_inn(config: Any,
              data_path: str = "./data"):

    exp_name = getattr(config, "exp_name")
    exp_path = Path(f"./experiments/{exp_name}/{str(time.time())}")
    try:
        exp_path.mkdir(parents=True, exist_ok=False)
        Path.joinpath(exp_path, "recons").mkdir(parents=True, exist_ok=True)
        Path.joinpath(exp_path, "latents").mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        print("Experiment already exists.")
        # sys.exit()

    dataset = config.dataset
    model_name = getattr(config, 'model', 'small')
    do_fwd = getattr(config, "do_fwd", True)
    do_rev = getattr(config, "do_rev", False)
    input_noise_sigma = getattr(config, "input_noise_sigma", 0.1)
    latent_noise_sigma = getattr(config, "latent_noise_sigma", 1e-2)
    pre_low_lr = getattr(config, "pre_low_lr", 0)
    # scale_l_fwd = getattr(config, "scale_l_fwd", False)
    n_epochs = getattr(config, "n_epochs", 1)
    n_its_per_epoch = getattr(config, "n_its_per_epoch", 2**16)
    ica = getattr(config, "ica", True)
    n_batches_ica_init = getattr(config, "n_batches_ica_init", 40)
    # burn_in_ica = getattr(config, "burn_in_ica", 1)
    empirical_var = getattr(config, "empirical_var", True)
    # n_blocks = getattr(config, "n_blocks", 24)
    # img_dims = getattr(config, "img_dims", (28, 28))
    coutpling_block = getattr(config, "coupling_block", "gin")
    internal_width = getattr(config, "internal_width", 512)
    # dropout = getattr(config, "dropout", 0.3)
    clamp = getattr(config, "clamp", 1.5)
    n_blocks_fc = getattr(config, "n_blocks_fc", 8)
    depths = getattr(config, "depths", [4, 6, 6, 6])
    channels = getattr(config, "channels", [32, 64, 128, 256])
    splits = getattr(config, "splits", False)
    # reshapes = getattr(config, "reshapes", ["reshape", "reshape", "reshape", "haar"])
    # kernel_size = getattr(config, "kernel_size", 1)
    init_scale = getattr(config, "init_scale", 0.03)
    conditional = getattr(config, "conditional", False)
    # cond_size = getattr(config, "cond_size", None)
    # cond_feature_channels = getattr(config, "cond_feature_channels", 256)
    # fc_cond_length = getattr(config, "fc_cond_length", 128)
    lr = getattr(config, "lr", 5e-4)
    betas = getattr(config, "betas", (0.9, 0.999))
    weight_decay = getattr(config, "weight_decay", 1e-5)
    batch_size = getattr(config, "batch_size", 32)
    sampling_temperature = getattr(config, "sampling_temperature", 1.)
    # decay_by = getattr(config, "decay_by", 0.01)
    live_visualization = getattr(config, "live_visualization", True)
    progress_bar = getattr(config, "progress_bar", True)
    checkpoint_save_interval = getattr(config, "checkpoint_save_interval", 120 * 3)
    from_checkpoint = getattr(config, "from_checkpoint", False)
    milestones = getattr(config, "milestones", [50, 100])

    exp_summary(save_path=str(exp_path), **locals())

    # data_path = './data' if data_path is None else data_path
    Path(f"{data_path}/{dataset}").mkdir(exist_ok=True, parents=True)
    train_data = getattr(data, f"{dataset}_train_data")(f"{data_path}/{dataset}")
    # if getattr(config, "make_class_cond", False):
    #     make_class_cond = getattr(data, f"{dataset}_make_class_cond", None)
    # else:
    #     make_class_cond = None
    img_dims = getattr(data, f"{dataset}_img_dims")

    # _img_dims = (2, *img_dims) if model_name == "conv" and len(img_dims) < 3 else img_dims
    n_classes = getattr(data, f"{dataset}_n_classes")
    model = getattr(models, f"{model_name}_model")(# n_blocks=n_blocks,
                                                   coutpling_block=coutpling_block,
                                                   ica=ica,
                                                   n_classes=n_classes,
                                                   n_blocks_fc=n_blocks_fc,
                                                   img_dims=img_dims,
                                                   depths=depths,
                                                   channels=channels,
                                                   splits=splits,
                                                   # reshapes=reshapes,
                                                   # kernel_size=kernel_size,
                                                   internal_width=internal_width,
                                                   # dropout=dropout,
                                                   clamp=clamp,
                                                   init_scale=init_scale,
                                                   # cond_size=cond_size,
                                                   conditional=conditional,)
                                                   # cond_feature_channels=cond_feature_channels,
                                                   # fc_cond_length=fc_cond_length)

    cond_sizes = copy.copy(model.cond_sizes)
    data_loader_kwargs = dict(batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)
    # if torch.cuda.device_count() > 1:
    #     torch.distributed.init_process_group(backend="nccl")
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None)
        # data_loader_kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count())
    # else:
    data_loader_kwargs["shuffle"] = True
    data_loader = torch.utils.data.DataLoader(train_data, **data_loader_kwargs)

    params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optim = torch.optim.Adam(params_trainable, lr=lr, betas=betas, eps=1e-6, weight_decay=weight_decay)

    if from_checkpoint:
        model_path = Path(f"./experiments/{exp_name}/{config.checkpoint_timestamp}/checkpoint.pt")# Path.joinpath(exp_path, "checkpoint.pt")
        if model_path.exists():
            checkpoint = torch.load(model_path)
            try:
               model.load_state_dict(checkpoint["model"])
            except RuntimeError:
                model.load_state_dict({'.'.join(key.split('.')[1:]): value for key, value in checkpoint["model"].items()})
            optim.load_state_dict(checkpoint["optim"])

    # gamma = (decay_by)**(1./n_epochs)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones)
    if live_visualization:
        live_viz = viz.LiveVisualizer(loss_labels=["L_fwd", "L_rev"],
                                      train_loader=data_loader,
                                      n_imgs=10,
                                      img_width=img_dims[-1],# getattr(data, f"{dataset}_img_width"),
                                      preview_upscale=max(int(100 / img_dims[-1]), 1),
                                      n_its_per_epoch=n_its_per_epoch)

    latent_pca = LatentPCA(model)

    losses = np.zeros((n_epochs, 2))

    for i_epoch in range(-pre_low_lr, n_epochs):

        loss_history = []
        zs, all_labels = [], []

        # if i_epoch < 0:
        #     for param_group in optim.param_groups:
        #         param_group['lr'] = lr * 2e-2

        if ica and empirical_var:
            model.init_mu_sig(data_loader, n_batches=n_batches_ica_init)

        for i_batch, (x, labels) in tqdm.tqdm(enumerate(data_loader),
                                             total=min(len(data_loader), n_its_per_epoch),
                                             leave=False,
                                             mininterval=1.,
                                             disable=(not progress_bar),
                                             ncols=83,
                                             desc=f"Epoch: {i_epoch}"):

            # first check that std will be well defined
            # if min([sum(labels == i).item() for i in range(n_classes)]) < 3:
            #     # don't calculate loss and update weights -- it will give nan or error
            #     # go to next batch
            #     continue
            if i_batch == n_its_per_epoch:
                break
            x, labels = x.cuda(), labels.cuda()

            if conditional:
                cond = data.make_cond(labels, cond_sizes, batch_size, n_classes, False if dataset != "celeba" else True)
            else:
                cond = None

            # _ica = False if i_epoch < burn_in_ica else ica
            (l_fwd, l_rev), x_rec, z = train_step(model=model,
                                                  optim=optim,
                                                  x=x,
                                                  y=labels,
                                                  n_classes=n_classes,
                                                  ica=ica,
                                                  empirical_var=empirical_var,
                                                  cond=cond,
                                                  do_fwd=do_fwd,
                                                  do_rev=do_rev,
                                                  input_noise_sigma=input_noise_sigma,
                                                  latent_noise_sigma=latent_noise_sigma)
                                                  # scale_l_fwd=scale_l_fwd)

            loss_history.append([l_fwd.item(), l_rev.item()])
            z = [_z.cpu() for _z in z] if type(z) == list else z.cpu()
            zs.append(z)
            all_labels.append(labels)

        zs = [torch.cat([_z[i] for _z in zs], dim=0) for i in range(len(zs[0]))] if type(zs[0]) == list else torch.cat(zs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        if lr_scheduler.get_last_lr()[0] > 1e-8:
            lr_scheduler.step()

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        if i_epoch >= 0:
            losses[i_epoch] = epoch_losses
        # epoch_losses[0] = min(epoch_losses[0], 0)

        print(f"E: {i_epoch} | L_fwd: {epoch_losses[0]:.4f} | L_rev: {epoch_losses[1]:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.7f}")

        with torch.no_grad():

            # output, _ = model(x.cuda(), cond)
            if isinstance(z, (tuple, list)):
                samples = [sample_outputs(o.size(), sampling_temperature) for o in z]
            else:
                samples = sample_outputs((1 if ica else batch_size, np.prod(img_dims)), sampling_temperature)

            if ica:
                model.init_mu_sig(data_loader, n_batches=n_batches_ica_init)
                samples = samples * model.log_sig.exp() + model.mu

            rev_imgs, _ = model(samples, cond, rev=True)

        if type(zs) == list:
            zs = zs[-1]
        # z_pca = latent_pca.fit_transform(zs)

        idxs = np.random.choice(rev_imgs.shape[0], 100)
        imgs_plot = rev_imgs[idxs].cpu()
        imgs_plot = [getattr(data, f"{dataset}_unnormalize")(img) for img in imgs_plot]

        if ica:
            sig_rms = np.sqrt(np.mean((model.log_sig.exp() ** 2).detach().cpu().numpy(), axis=0))
            sig_rms_sorted_idx = np.argsort(sig_rms)[::-1]
            sig_rms_sorted = sig_rms[sig_rms_sorted_idx]

        if live_visualization:
            live_viz.update_losses(epoch_losses, logscale=False)
            live_viz.update_images(*imgs_plot)
            # latent_idx = np.random.choice(z_pca.shape[0], min(500, z_pca.shape[0]))
            # live_viz.update_latent(z_pca[latent_idx], labels[latent_idx])
            if ica:
                live_viz.update_sig_rms(sig_rms_sorted)

        if i_epoch % checkpoint_save_interval == 0 and i_epoch >= 0: # and i_epoch > 0:
            torch.save(dict(model=model.state_dict(), optim=optim.state_dict()), Path.joinpath(exp_path, "checkpoint.pt"))

            fig, ax = plt.subplots()
            ax.plot(range(i_epoch + 1), losses[:,0][:i_epoch + 1], label="L_fwd")
            ax.plot(range(i_epoch + 1), losses[:,1][:i_epoch + 1], label="L_rev")
            ax.legend()
            plt.tight_layout()
            fig.savefig(Path.joinpath(exp_path, "losses.png"), bbox_inches="tight")

            show_img = viz.grid_img(imgs_plot, n_imgs=10)
            fig, ax = plt.subplots()
            ax.imshow(np.moveaxis(show_img, 0, -1), cmap="gray", vmin=0., vmax=1.)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            fig.savefig(Path.joinpath(exp_path, f"recons/{i_epoch}.png"), bbox_inches="tight")

            # latent_idx = np.random.choice(z_pca.shape[0], min(500, z_pca.shape[0]))
            # _labels = labels[latent_idx].cpu()
            # _labels -= _labels.min()

            # fig, ax = plt.subplots()
            # if len(_labels.shape) == 1:
            #     for l in np.unique(_labels.numpy()):
            #         ax.scatter(z_pca[latent_idx][_labels == l][:,0], z_pca[latent_idx][_labels == l][:,1], label=l)
            # else:
            #     ax.scatter(z_pca[latent_idx][:, 0], z_pca[latent_idx][:, 1])
            # ax.legend()
            # plt.tight_layout()
            # fig.savefig(Path.joinpath(exp_path, f"latents/{i_epoch}.png"), bbox_inches="tight")

            plt.close('all')

if __name__ == "__main__":
    import importlib
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--data_path", "-d", type=str)
    args = parser.parse_args()

    # np.seterr(all='raise')

    config = importlib.import_module(f"configs.{args.config}")

    train_inn(config=config, data_path=args.data_path)