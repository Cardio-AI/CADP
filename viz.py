from typing import List, Tuple, Union, Optional
import torch
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
from typing import Any
from ipywidgets import interact, IntSlider
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import visdom
import FrEIA.framework as ff

def plot_img_batch(img_batch: torch.Tensor,
                   n_rows: int = 1,
                   n_cols: int = None,
                   figsize: Tuple[int, int] = (20, 20)) -> plt.Figure:
    if n_cols is None:
        fig, ax = plt.subplots(figsize=figsize)
        img = torch.cat([i for i in img_batch], dim=-1)
        if img.size(0) == 2:
            img = torch.mean(img, dim=0)
        elif img.size(0) == 3:
            img = np.moveaxis(img.numpy(), 0, -1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig, axs = plt.subplots(n_rows, n_cols)
        idx = np.random.choice(len(img_batch), (n_rows*n_cols,)).tolist()
        # TODO: handle 1 and 3 channels
        for i, img in enumerate(img_batch[idx]):
            ax = axs[i % n_rows,i % n_cols]
            if img.size(0) == 2:
                img = torch.mean(img, dim=0)
            elif img.size(0) == 3:
                img = np.moveaxis(img.numpy(), 0, -1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    return fig

def scatter_plot_latent_space(z_pca: Union[np.ndarray, List[np.ndarray]],
                              labels: torch.Tensor,
                              every_nth_data_point: int = 5,
                              figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    if type(z_pca) == np.ndarray:
        z_pca = [z_pca]
    fig, axs = plt.subplots(1, len(z_pca), figsize=figsize)
    lims = []
    if len(z_pca) == 1:
        axs = [axs]
    for ax, z in zip(axs, z_pca):
        for l in torch.unique(labels):
            ax.scatter(z[labels.cpu() == l][:, 0][::every_nth_data_point],
                       z[labels.cpu() == l][:, 1][::every_nth_data_point],
                       label=l.item())
            ax.set_aspect("equal")
            lims += ax.get_xlim()
            lims += ax.get_ylim()
    lims = [np.min(lims), np.max(lims)]
    for ax in axs:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_latent_space_2d(zs: Union[torch.Tensor, Tuple[torch.Tensor,...]],
                         labels: torch.Tensor,
                         every_nth_data_point: int = 5) -> plt.Figure:
    if type(zs) == torch.Tensor:
        zs = [zs]
    z_pca = []
    for z in zs:
        pca = PCA(n_components=2, whiten=True)
        _z_pca = pca.fit_transform(z)
        z_pca.append(_z_pca)
    return scatter_plot_latent_space(z_pca, labels, every_nth_data_point)

def grid(model: ff.GraphINN,
         zs: torch.Tensor,
         cond: Optional[torch.Tensor] = None,
         n_pts: int = 20,
         img_dims: Tuple[int,...] = (28, 28),
         start: int = -3,
         stop: int = 3) -> np.ndarray:
    h, w = img_dims[-2], img_dims[-1]
    x = np.linspace(start, stop, n_pts)[None]
    xx, yy = np.meshgrid(x, x)
    grid = np.concatenate([xx[..., None], yy[..., None]], axis=-1).reshape(-1, 2)

    if type(zs) == tuple:
        z_pca_backprojected = []
        for z in zs:
            pca = PCA(n_components=2, whiten=True)
            z_pca = pca.fit_transform(z.cpu().detach())
            _z_pca_backprojected = pca.inverse_transform(grid)
            _z_pca_backprojected = torch.from_numpy(_z_pca_backprojected).float().cuda()
            z_pca_backprojected.append(_z_pca_backprojected)
        z_pca_backprojected = tuple(z_pca_backprojected)
    else:
        pca = PCA(n_components=2, whiten=True)
        z_pca = pca.fit_transform(zs.cpu().detach())
        z_pca_backprojected = pca.inverse_transform(grid)
        z_pca_backprojected = torch.from_numpy(z_pca_backprojected).float().cuda()
    with torch.no_grad():
        try:
            rev, _ = model(z_pca_backprojected, cond, rev=True)
        except RuntimeError:
            rev = torch.cat([model(z_pca_backprojected[i:i+1], cond[i:i+1], rev=True)[0] for i in range(cond.shape[0])], dim=0)
    rev = rev.view(n_pts, n_pts, *img_dims).cpu()

    if len(img_dims) == 2:
        all_imgs = torch.zeros(n_pts * h, n_pts * w)
        for i in range(n_pts):
            for j in range(n_pts):
                all_imgs[i * h:(i + 1) * h, j * w:(j + 1) * w] = rev[i, j]
        # all_imgs = all_imgs.numpy()
    # elif len(img_dims) == 3:
    else:
        all_imgs = torch.zeros(img_dims[0], n_pts * h, n_pts * w)
        for i in range(n_pts):
            for j in range(n_pts):
                all_imgs[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = rev[i, j]
        # all_imgs = np.moveaxis(all_imgs.numpy(), 0, -1)
    return all_imgs

def grid_img(img_list, n_imgs: int = 10) -> np.ndarray:

    w = img_list[0].shape[1]
    k = 0
    # k_img = 0

    show_img = np.zeros((3, w * n_imgs, w * n_imgs), dtype=np.uint8)
    img_list_np = []
    for im in img_list:
        im_np = im.cpu().data.numpy()
        img_list_np.append(np.clip((255. * im_np), 0, 255).astype(np.uint8))

    for i in range(n_imgs):
        for j in range(n_imgs):
            _img = img_list_np[k]  # [k_img]
            if len(_img.shape) == 3 and _img.shape[0] != 3:
                _img = np.mean(_img, axis=0)

            show_img[:, w * i:w * i + w, w * j:w * j + w] = _img

            k += 1
            # if k >= len(img_list_np):
            #     k = 0
            #     k_img += 1
    return show_img

class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.counter = 0

            header = 'Epoch'
            for l in loss_labels:
                header += '\t\t%s' % (l)

            print(header)

    def update_losses(self, losses, *args):
        # print('\r', '    '*20, end='')
        # line = '\r%.3i' % (self.counter)
        # for l in losses:
        #     line += '\t\t%.4f' % (l)
        #
        # print(line)
        self.counter += 1

    def update_images(self, *args):
        pass

class LiveVisualizer(Visualizer):
    def __init__(self,
                 loss_labels: List[str],
                 train_loader: torch.utils.data.DataLoader,
                 n_imgs: int = 10,
                 img_width: int = 28,
                 preview_upscale: int = 3,
                 n_its_per_epoch: int = 2**16):
        super().__init__(loss_labels)
        self.viz = visdom.Visdom()#env='mnist')
        self.viz.close()

        self.l_plots = self.viz.line(X=np.zeros((1,self.n_losses)),
                                     Y=np.zeros((1,self.n_losses)),
                                     opts={'legend':self.loss_labels})

        self.imgs = self.viz.image(np.random.random((3, img_width*n_imgs*preview_upscale,
                                                        img_width*n_imgs*preview_upscale)))

        # self.latent = self.viz.scatter(X=np.zeros((1000,2)))

        self.sig_rms = self.viz.line(X=np.zeros((1,)), Y=np.zeros((1,)))

        self.train_loader = train_loader
        self.n_its_per_epoch = n_its_per_epoch

        self.n_imgs = n_imgs
        self.preview_upscale = preview_upscale

    def update_losses(self, losses, logscale=True):
        super().update_losses(losses)
        its = min(len(self.train_loader), self.n_its_per_epoch)
        y = np.array([losses])
        if logscale:
            y = np.log10(y)

        self.viz.line(X=(self.counter-1) * its * np.ones((1,self.n_losses)),
                      Y=y,
                      opts={'legend':self.loss_labels},
                      win=self.l_plots,
                      update='append')

    def update_images(self, *img_list):

        show_img = grid_img(img_list, self.n_imgs)

        show_img = zoom(show_img, (1., self.preview_upscale, self.preview_upscale), order=0)

        self.viz.image(show_img, win=self.imgs)

    def update_latent(self, zs: torch.Tensor, labels: torch.Tensor):
        cp = np.array(sns.color_palette())
        cp = (cp * 255).astype(np.uint8)
        labels -= labels.min()
        opts = dict(markercolor=cp[labels.tolist()]) if len(labels.shape) == 1 else dict()
        self.viz.scatter(X=zs,
                         # Y=labels + 1,
                         win=self.latent,
                         update="replace",
                         opts=opts)

    def update_sig_rms(self, sig_rms: np.ndarray):
        self.viz.line(X=np.arange(sig_rms.shape[0]),
                      Y=sig_rms,
                      win=self.sig_rms,
                      update="replace")

    def close(self):
        self.viz.close(win=self.imgs)
        self.viz.close(win=self.l_plots)
        self.viz.close(win=self.latent)

def interactive_display_dataset(
        dataset: Dataset,
        label_overlay: bool = False,
        dims: str = "2d"
) -> Any:
    # z_slider = IntSlider(min=0, max=dataset[0][0].shape[1] - 1) if dims == "3d" else None
    @interact
    def display(img=IntSlider(min=0, max=len(dataset) - 1)):#, z=z_slider):
        def check_size(img: torch.Tensor) -> torch.Tensor:
            if img.size(0) == 3:
                img = torch.moveaxis(img, 0, -1)
            elif img.size(0) == 1:
                img = img[0]
            return img

        x, y = dataset[img]

        x = check_size(x)
        #if dims == "3d":
        #    x = x[z]
        #    if label_overlay:
        #        y = check_size(y)
        #        y = y[z]
        fig, ax = plt.subplots()
        ax.imshow(x, cmap="gray")
        #if label_overlay:
        #    ax.imshow(y, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    return display