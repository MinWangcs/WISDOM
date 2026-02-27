"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import pywt
import math

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,

            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    Policy network
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):

    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class Y_Network(nn.Module):
    """ 
    Wavelet representation network 
    """
    def __init__(self, d_model, kernel_size=None, depth=None, wavelet_init=None,
                 seq_len=None, dropout=0., indep_res_init=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.seq_len = seq_len
        if depth is not None:
            self.depth = depth
        elif seq_len is not None:
            self.depth = self.max_depth(seq_len)
        else:
            raise ValueError("Provide depth or seq_len.")

        self.m = self.depth + 1

        with torch.no_grad():
            if wavelet_init is not None:
                self.wavelet = pywt.Wavelet(wavelet_init)
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
                self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [d_model, 1, 1]))
                self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [d_model, 1, 1]))
            elif kernel_size is not None:
                self.h0 = nn.Parameter(
                    torch.empty(d_model, 1, kernel_size).uniform_(-1., 1.) * 
                    math.sqrt(2.0 / (kernel_size * 2))
                )
                self.h1 = nn.Parameter(
                    torch.empty(d_model, 1, kernel_size).uniform_(-1., 1.) * 
                    math.sqrt(2.0 / (kernel_size * 2))
                )
            else:
                raise ValueError("Initialize the kernel.")

            w_init = torch.empty(
                d_model, self.m + 1).uniform_(-1., 1.) * math.sqrt(2.0 / (2*self.m + 2))
            if indep_res_init:
                w_init[:, -1] = torch.empty(d_model).uniform_(-1., 1.)
            self.w = nn.Parameter(w_init)      # weights of the linear layer

        self.activation = nn.GELU()
        dropout_fn = nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0. else nn.Identity()

    def max_depth(self, L):
        depth = math.ceil(math.log2((L - 1) / (self.kernel_size - 1) + 1))
        return depth

    def forward(self, x):
        y, res_lo = forward_fading(x, self.h0, self.h1, self.w, self.depth, self.kernel_size)
        y = self.dropout(self.activation(y))
        return y, res_lo


def forward_fading(x, h0, h1, w, depth, kernel_size):
    res_lo = x
    y = 0.
    dilation = 1
    for i in range(depth, 0, -1):      # depth: number of decompositions
        padding = dilation * (kernel_size - 1)
        res_lo_pad = torch.nn.functional.pad(res_lo, (padding, 0), "constant", 0)
        # Compute detail coefficients
        res_hi = torch.nn.functional.conv1d(res_lo_pad, h1, dilation=dilation, groups=x.shape[1])
        # Compute approximation coefficients
        res_lo = torch.nn.functional.conv1d(res_lo_pad, h0, dilation=dilation, groups=x.shape[1])
        y += w[:, i:i + 1] * res_hi
        dilation *= 2

    y += w[:, :1] * res_lo
    y += x * w[:, -1:]
    return y, res_lo


