import pdb
from typing import Optional, Tuple, Iterable, List, Union
import functools
from math import exp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import FrEIA.framework as ff
import FrEIA.modules as fm

# def subnet_fc(c_in: int,
#               c_out: int,
#               internal_width: int = 392,
#               init_scale: float = 0.03,
#               leaky_slope: float = 0.01,
#               dropout: float = 0.) -> nn.Sequential:
#
#     subnet = nn.Sequential(nn.Linear(c_in, internal_width),
#                            nn.LeakyReLU(negative_slope=leaky_slope),
#                            nn.Dropout(p=dropout),
#                            nn.Linear(internal_width, internal_width),
#                            nn.LeakyReLU(negative_slope=leaky_slope),
#                            nn.Dropout(p=dropout),
#                            nn.Linear(internal_width,  c_out))
#     for l in subnet:
#         if isinstance(l, nn.Linear):
#             # nn.init.xavier_normal_(l.weight)
#             l.weight.data = init_scale * torch.randn(l.weight.shape)
#             l.bias.data = init_scale * torch.randn(l.bias.shape)
#             # l.weight.data.fill_(init_scale * torch.randn(l.weight.shape).cuda())
#             # l.bias.data.fill_(init_scale * torch.randn(l.bias.shape).cuda())
#     subnet[-1].weight.data.fill_(0.)
#     subnet[-1].bias.data.fill_(0.)
#     return subnet
#
# def subnet_conv(c_in: int,
#                 c_out: int,
#                 internal_width: int = 512,
#                 kernel_size: int = 3,
#                 init_scale: float = 0.03,
#                 leaky_slope: float = 0.01,
#                 dropout: float = 0.) -> nn.Sequential:
#     # width = 512
#     pad = kernel_size // 2
#     kernel_size = nn.modules.utils._pair(kernel_size)
#     subnet = nn.Sequential(nn.Conv2d(c_in, internal_width, kernel_size, padding=pad),
#                            nn.LeakyReLU(negative_slope=leaky_slope),
#                            nn.Dropout(p=dropout),
#                            nn.Conv2d(internal_width, internal_width, kernel_size, padding=pad),
#                            nn.LeakyReLU(negative_slope=leaky_slope),
#                            nn.Dropout(p=dropout),
#                            nn.Conv2d(internal_width, c_out, kernel_size, padding=pad))
#
#     for l in subnet:
#         if isinstance(l, nn.Conv2d):
#             # nn.init.xavier_normal_(l.weight)
#             l.weight.data = init_scale * torch.randn(l.weight.shape)
#             l.bias.data = init_scale * torch.randn(l.bias.shape)
#             # l.weight.data.fill_(init_scale * torch.randn(l.weight.shape).cuda())
#             # l.bias.data.fill_(init_scale * torch.randn(l.bias.shape).cuda())
#     subnet[-1].weight.data.fill_(0.)
#     subnet[-1].bias.data.fill_(0.)
#     return subnet

class ICAModel(nn.Module):
    def __init__(self,
                 inn: ff.ReversibleGraphNet,
                 n_classes: int,
                 n_dims: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_dims = n_dims

        self.inn = inn
        self.mu = nn.Parameter(torch.zeros(n_classes, n_dims).cuda()).requires_grad_()
        self.log_sig = nn.Parameter(torch.zeros(n_classes, n_dims).cuda()).requires_grad_()

    def forward(self, x: torch.Tensor, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inn(x, rev=rev)

    def init_mu_sig(self, data_loader: DataLoader, n_batches: int = 40) -> None:
        zs, labels = [], []
        for i, (x, y) in enumerate(data_loader):
            if i == n_batches:
                break
            x, y = x.cuda(), y.cuda()
            # if len(x.size()) == 4 and x.size(1) == 1:
            #     x = torch.cat([x, x], dim=1)
            with torch.no_grad():
                z, _ = self.inn(x)
            zs.append(z)
            labels.append(y)
        zs = torch.cat(zs, dim=0)
        labels = torch.cat(labels, dim=0)

        self.mu.data = torch.stack([zs[labels == i].mean(0) for i in range(self.n_classes)])
        self.log_sig.data = torch.stack([zs[labels == i].std(0) for i in range(self.n_classes)]).log()

def conv_model(img_dims: Tuple[int,...],
               ica: bool = False,
               n_classes: int = 2,
               coupling_block: str = "glow", # "gin"
               # n_blocks: Optional[int] = None,
               n_blocks_fc: int = 8,
               depths: Union[List[int], int] = [4, 6],
               channels: Union[List[int], int] = [128, 256],
               splits: Union[List[Union[float, bool]], Union[bool, float]] = False,
               # reshapes: Union[List[str], str] = ["reshape", "reshape", "reshape", "haar"],
               kernel_size: int = 1,
               internal_width: int = 64,
               dropout: float = 0.3,
               clamp: float = 2.0,
               init_scale: float = 0.03,
               conditional: bool = True,
               *args,
               **kargs) -> Tuple[ff.ReversibleGraphNet, List[Tuple[int,...]]]:

    # if isinstance(depths, int):
    #     depths = [depths] * n_blocks
    # # if isinstance(channels, int):
    # #     channels = [channels] * n_blocks
    # if isinstance(splits, (bool, float)):
    #     splits = [splits] * n_blocks
    # if isinstance(reshapes, str):
    #     reshapes = [reshapes] * n_blocks

    # def random_orthog(n):
    #     w = np.random.randn(n, n)
    #     w = w + w.T
    #     w, S, V = np.linalg.svd(w)
    #     return torch.FloatTensor(w)

    if coupling_block == "gin":
        coupling_block = fm.GINCouplingBlock
    else:
        coupling_block = fm.GLOWCouplingBlock

    if not splits:
        splits = [False] * len(depths)

    nodes = [ff.InputNode(*img_dims, name='inp')]
    cond_nodes, cond_sizes = [], []
    # #
    # # # channels was in subnet_conv
    # for i, (d, s, r) in enumerate(zip(depths, splits, reshapes)):
    #     if r == 'haar':
    #         nodes.append(ff.Node([nodes[-1].out0],
    #                              fm.HaarDownsampling,
    #                              {'rebalance': 0.5, 'order_by_wavelet': True},
    #                              name='haar'))
    #     elif r == 'reshape':
    #         nodes.append(ff.Node([nodes[-1].out0], fm.IRevNetDownsampling, {}, name='reshape'))
    #
    #     _subnet_conv = lambda dims_in, dims_out: subnet_conv(c_in=dims_in,
    #                                                          c_out=dims_out,
    #                                                          # internal_width=internal_width,
    #                                                          init_scale=init_scale,
    #                                                          dropout=dropout,
    #                                                          kernel_size=kernel_size)
    #     for k in range(d):
    #         nodes.append(ff.Node([nodes[-1].out0],
    #                              fm.Fixed1x1Conv,
    #                              {'M': random_orthog(nodes[-1].out0[0].output_dims[0][0])},
    #                              name=f"1x1_{d}_{k}"))
    #         nodes.append(ff.Node([nodes[-1].out0],
    #                              coupling_block,
    #                              {"clamp": clamp, "subnet_constructor": _subnet_conv},
    #                              name=f"conv_{d}_{k}"))
    #         # nodes.append(Ff.Node(nodes[-1],
    #         #                     Fm.PermuteRandom,
    #         #                     {'seed':np.random.randint(2**31)},
    #         #                     name=F'permute_conv_{d}_{k}'))
    #
    #     if s:
    #         out_ch = nodes[-1].out0[0].output_dims[0][0]
    #         section_sizes = (int(out_ch * s), int(out_ch * (1 - s)))
    #         nodes.append(ff.Node([nodes[-1].out0],
    #                              fm.Split,
    #                              {'section_sizes': section_sizes, 'dim': 0},
    #                              name=f'split_{i}'))
    #         output = ff.Node([nodes[-1].out1], fm.Flatten, {}, name='flatten')
    #         nodes.insert(-2, output)
    #         nodes.insert(-2, ff.OutputNode([output.out0], name=f'out_{i}'))
    #
    # nodes.append(ff.Node([nodes[-1].out0], fm.Flatten, {}, name='flatten'))
    # for k in range(n_blocks_fc):
    #     nodes.append(ff.Node([nodes[-1].out0], fm.PermuteRandom, {'seed': k}, name=F'permute_{k}'))
    #     nodes.append(ff.Node([nodes[-1].out0],
    #                          coupling_block,
    #                          {'clamp': clamp, 'subnet_constructor': subnet_fc},
    #                          name=F'fc_{k}'))
    #
    # nodes.append(ff.OutputNode([nodes[-1].out0], name='out'))

    ######################################

    def subnet_fc(c_in: int, c_out: int, internal_width: int) -> nn.Sequential:
        # width = 1024
        subnet = nn.Sequential(nn.Linear(c_in, internal_width), nn.ReLU(),
                               nn.Linear(internal_width, internal_width), nn.ReLU(),
                               nn.Linear(internal_width, c_out))
        for l in subnet:
            if isinstance(l, nn.Linear):
                l.weight.data = init_scale * torch.randn_like(l.weight.data)
                l.bias.data = init_scale * torch.randn_like(l.bias.data)
                # nn.init.xavier_normal_(l.weight)
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
        return subnet

    def subnet_conv(c_in: int, c_out: int, internal_width: int) -> nn.Sequential:
        # width = 128
        subnet = nn.Sequential(nn.Conv2d(c_in, internal_width, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(internal_width, internal_width, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(internal_width, c_out, 3, padding=1))
        for l in subnet:
            if isinstance(l, nn.Conv2d):
                # nn.init.xavier_normal_(l.weight)
                l.weight.data = init_scale * torch.randn_like(l.weight.data)
                l.bias.data = init_scale * torch.randn_like(l.bias.data)
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
        return subnet

    # def subnet_conv2(c_in, c_out):
    #     width = 256
    #     subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
    #                            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
    #                            nn.Conv2d(width, c_out, 3, padding=1))
    #     for l in subnet:
    #         if isinstance(l, nn.Conv2d):
    #             nn.init.xavier_normal_(l.weight)
    #     subnet[-1].weight.data.fill_(0.)
    #     subnet[-1].bias.data.fill_(0.)
    #     return subnet

    # nodes.append(ff.Node(nodes[-1], fm.IRevNetDownsampling, {}, name='downsample0'))

    for i, (d, ch, s) in enumerate(zip(depths, channels, splits)):
        nodes.append(ff.Node(nodes[-1], fm.IRevNetDownsampling, {}, name=f'downsample{i}'))

        out_dims, = nodes[-1].out0[0].output_dims
        cond_size = (n_classes, out_dims[-2], out_dims[-1])
        cond_node = ff.ConditionNode(*cond_size) if conditional else None
        cond_sizes.append(cond_size)
        cond_nodes.append(cond_node)
        for k in range(d):
            nodes.append(ff.Node(nodes[-1], coupling_block,
                                 {'subnet_constructor': lambda c_in, c_out: subnet_conv(c_in, c_out, 128), 'clamp': clamp},
                                 conditions=cond_node,
                                 name=F'coupling_conv{i}_{k}'))
            nodes.append(ff.Node(nodes[-1],
                                 fm.PermuteRandom,
                                 {'seed': np.random.randint(2 ** 31)},
                                 name=F'permute_conv{i}_{k}'))

        if s:
            out_ch = nodes[-1].out0[0].output_dims[0][0]
            section_sizes = (int(out_ch * s), int(out_ch * (1 - s)))
            nodes.append(ff.Node([nodes[-1].out0],
                                 fm.Split,
                                 {'section_sizes': section_sizes, 'dim': 0},
                                 name=f'split_{i}'))
            output = ff.Node([nodes[-1].out1], fm.Flatten, {}, name='flatten')
            nodes.insert(-2, output)
            nodes.insert(-2, ff.OutputNode([output.out0], name=f'out_{i}'))

        # nodes.append(ff.Node(nodes[-1], fm.IRevNetDownsampling, {}, name=f'downsample{i+1}'))

    nodes.append(ff.Node(nodes[-1], fm.Flatten, {}, name='flatten'))

    cond_node = ff.ConditionNode(n_classes) if conditional else None
    cond_nodes.append(cond_node)
    cond_sizes.append((n_classes,))

    for i, k in enumerate(range(n_blocks_fc)):
        nodes.append(ff.Node(nodes[-1], coupling_block,
                             {'subnet_constructor': lambda c_in, c_out: subnet_fc(c_in, c_out, internal_width), 'clamp': clamp},
                             conditions=cond_node,
                             name=F'coupling_fc{i}'))
        nodes.append(ff.Node(nodes[-1],
                             fm.PermuteRandom,
                             {'seed': np.random.randint(2 ** 31)},
                             name=F'permute_fc{i}'))

    nodes.append(ff.OutputNode(nodes[-1], name='output'))

    nodes += list(filter(lambda x: x is not None, cond_nodes))

    model = ff.ReversibleGraphNet(nodes, verbose=False)
    model = model.cuda()

    model.cond_sizes = cond_sizes

    if ica:
        model = ICAModel(inn=model, n_classes=n_classes, n_dims=np.prod(img_dims))
    return model #, cond_sizes

def fc_model(n_blocks: int,
             img_dims: Tuple[int,...],
             internal_width: int = 64,
             dropout: float = 0.3,
             clamp: float = 1.5,
             init_scale: float = 0.03,
             # conditional: bool = False,
             # cond_size: Optional[Tuple[int,...]] = None,
             *args,
             **kwargs) -> ff.ReversibleGraphNet:

    # subnet_fc = functools.partial(SubnetFC, internal_width=internal_width, dropout=dropout)
    _subnet_fc = lambda dims_in, dims_out: subnet_fc(c_in=dims_in,
                                                     c_out=dims_out,
                                                     internal_width=internal_width,
                                                     dropout=dropout,
                                                     init_scale=init_scale)
    # def init_model(mod):
    #     for key, param in mod.named_parameters():
    #         split = key.split('.')
    #         if param.requires_grad:
    #             param.data = init_scale * torch.randn(param.data.shape).cuda()
    #             if split[3][-1] == '9': # last convolution in the coeff func
    #                 param.data.fill_(0.)

    nodes = [ff.InputNode(*img_dims, name='inp')]

    # cond_node = None if not conditional else ff.ConditionNode(cond_size)

    nodes.append(ff.Node([nodes[-1].out0], fm.Flatten, {}, name='flatten'))
    for i in range(n_blocks):
        nodes.append(ff.Node([nodes[-1].out0], fm.PermuteRandom, {'seed':i}, name=F'permute_{i}'))
        nodes.append(ff.Node([nodes[-1].out0],
                             fm.GLOWCouplingBlock,
                             {'clamp':clamp, 'subnet_constructor': _subnet_fc},
                             # conditions=cond_node,
                             name=F'fc_{i}'))

    nodes.append(ff.OutputNode([nodes[-1].out0], name='out'))
    # if conditional:
    #     nodes.append(cond_node)

    model = ff.ReversibleGraphNet(nodes, verbose=False)
    # init_model(model)
    model = model.cuda()

    return model
