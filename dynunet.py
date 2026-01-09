# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

#from memory_profiler import profile

from collections.abc import Sequence
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock #, UnetUpBlock
import numpy as np

# python library for added modules
from torch import einsum
import torch.nn.functional as F
from monai.utils import optional_import
from monai.networks.blocks.mlp import MLPBlock
#from monai.networks.blocks.selfattention import SABlock
from monai.networks.layers.factories import Act, Norm
from monai.networks.blocks.convolutions import Convolution
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

__all__ = ["DynUNet", "DynUnet", "Dynunet"]
from LanguageCrossAttention import LangCrossAtt


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    act: tuple | str | None = Act.PRELU,
    norm: tuple | str | None = Norm.INSTANCE,
    dropout: tuple | str | float | None = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Sequence[int] | int, stride: Sequence[int] | int, padding: Sequence[int] | int
) -> tuple[int, ...] | int:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def zero_pad_and_cat(tensor1, tensor2, dim=1):
    # Get the shapes of the tensors
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    # Determine the required padding for each dimension
    padding = []
    for i in range(len(shape1) - 1, 1, -1):
        size_diff = shape2[i] - shape1[i]
        if size_diff != 0:
            if size_diff > 0:
                # tensor1 needs padding
                padding.extend([0, size_diff])
            else:
                # tensor2 needs padding
                padding.extend([-size_diff, 0])
        else:
            padding.extend([0, 0])

    # Reverse padding list to match F.pad requirement
    padding = padding[::-1]
    # Apply zero padding to tensor1 if needed
    if any(padding):
        tensor1 = F.pad(tensor1, padding)

    # Apply zero padding to tensor2 if needed (reverse the padding for tensor2)
    if any(padding[::-1]):
        tensor2 = F.pad(tensor2, padding[::-1])

    # Concatenate the tensors along the specified dimension
    return torch.cat((tensor1, tensor2), dim=dim)


class UnetUpBlock(nn.Module):
    """
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )
        
        lang_dimension = 768
        self.lang_att1 = LangCrossAtt(emb_dim=192)
        self.lang_proj1 = nn.Linear(lang_dimension, 192)
        self.lang_att2 = LangCrossAtt(emb_dim=128)
        self.lang_proj2 = nn.Linear(lang_dimension, 128)
        

        self.lang_att3 = LangCrossAtt(emb_dim=96)
        self.lang_proj3 = nn.Linear(lang_dimension, 96)
        
        #self.lang_att4 = LangCrossAtt(emb_dim=64)
        #self.lang_proj4 = nn.Linear(lang_dimension, 64)

        #self.LangCrossAtt(emb_dim=192)

    def forward(self, inp, skip, lang_rep):
        # number of channels for skip should equals to out_channels
        #print(f"size of inp: {inp.size()}")
        #print(f"skip size: {skip.size()}")
        out = self.transp_conv(inp)         #does the upsampling
        #print(f"size after convolutions: {out.size()}")
        
        if out.size()[1] == 192:
            lang_rep1 = self.lang_proj1(lang_rep)
            out = self.lang_att1(lang_rep = lang_rep1, vision_rep = out)
        if out.size()[1] == 128:
            lang_rep2 = self.lang_proj2(lang_rep)
            out = self.lang_att2(lang_rep = lang_rep2, vision_rep = out)
        
        if out.size()[1] == 96:
            lang_rep3 = self.lang_proj3(lang_rep)
            out = self.lang_att3(lang_rep = lang_rep3, vision_rep = out)
        

        #if out.size()[1] == 64:
        #    lang_rep4 = self.lang_proj4(lang_rep)
        #    out = self.lang_att4(lang_rep = lang_rep4, vision_rep = out)
        #print(f"size of lang rep where attention is calculated: {lang_rep.size()}")
        #print("most likely where I apply attention")
        #print(f"Shape of out: {out.shape}")
        #print(f"Shape of skip: {skip.shape}")

        #out = zero_pad_and_cat(out, skip, dim=1)
        #print(f"Shape of out after: {out.shape}")

        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        #print(f"upblock size: {out.size()}")
        return out




class DynUNetSkipLayer(nn.Module):
    """
    Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip connection.
    The member `next_layer` may refer to instances of this class or the final bottleneck layer at the bottom the UNet
    structure. The purpose of using a recursive class like this is to get around the Torchscript restrictions on
    looping over lists of layers and accumulating lists of output tensors which must be indexed. The `heads` list is
    shared amongst all the instances of this class and is used to store the output from the supervision heads during
    forward passes of the network.
    """

    heads: list[torch.Tensor] | None

    def __init__(self, index, downsample, upsample, next_layer, heads=None, super_head=None):
        super().__init__()
        self.downsample = downsample
        self.next_layer = next_layer
        self.upsample = upsample
        self.super_head = super_head
        self.heads = heads
        self.index = index

    def forward(self, x, lang_rep):
        #print(f"x size: {x.size()}")
        #print(f"lang_rep size: {lang_rep.size()}")
        downout = self.downsample(x)
        #print(f"downsample type: {type(self.downsample)}")
        #print(downout.size()[1])
        #print(f"downsample: {downout.size()}")
        if downout.size()[1] == 192:
            nextout = self.next_layer(downout)
        else:
            nextout = self.next_layer(downout, lang_rep)
        #print(f"nextout size: {nextout.size()}")
        upout = self.upsample(nextout, downout, lang_rep)
        #print(f"upout size: {upout.size()}")
        if self.super_head is not None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout)

        return upout


class DynUNet(nn.Module):
    """
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        upsample_kernel_size: convolution kernel size for transposed convolution layers. The values should
            equal to strides[1:].
        filters: number of output channels for each blocks. Different from nnU-Net, in this implementation we add
            this argument to make the network more flexible. As shown in the third reference, one way to determine
            this argument is like:
            ``[64, 96, 128, 192, 256, 384, 512, 768, 1024][: len(strides)]``.
            The above way is used in the network that wins task 1 in the BraTS21 Challenge.
            If not specified, the way which nnUNet used will be employed. Defaults to ``None``.
        dropout: dropout ratio. Defaults to no dropout.
        norm_name: feature normalization type and arguments. Defaults to ``INSTANCE``.
            `INSTANCE_NVFUSER` is a faster version of the instance norm layer, it can be used when:
            1) `spatial_dims=3`, 2) CUDA device is available, 3) `apex` is installed and 4) non-Windows OS is used.
        act_name: activation layer type and arguments. Defaults to ``leakyrelu``.
        deep_supervision: whether to add deep supervision head before output. Defaults to ``False``.
            If ``True``, in training mode, the forward function will output not only the final feature map
            (from `output_block`), but also the feature maps that come from the intermediate up sample layers.
            In order to unify the return type (the restriction of TorchScript), all intermediate
            feature maps are interpolated into the same size as the final feature map and stacked together
            (with a new dimension in the first axis)into one single tensor.
            For instance, if there are two intermediate feature maps with shapes: (1, 2, 16, 12) and
            (1, 2, 8, 6), and the final feature map has the shape (1, 2, 32, 24), then all intermediate feature maps
            will be interpolated into (1, 2, 32, 24), and the stacked tensor will has the shape (1, 3, 2, 32, 24).
            When calculating the loss, you can use torch.unbind to get all feature maps can compute the loss
            one by one with the ground truth, then do a weighted average for all losses to achieve the final loss.
        deep_supr_num: number of feature maps that will output during deep supervision head. The
            value should be larger than 0 and less than the number of up sample layers.
            Defaults to 1.
        res_block: whether to use residual connection based convolution blocks during the network.
            Defaults to ``False``.
        trans_bias: whether to set the bias parameter in transposed convolution layers. Defaults to ``False``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Sequence[int] | int],
        strides: Sequence[Sequence[int] | int],
        upsample_kernel_size: Sequence[Sequence[int] | int],
        filters: Sequence[int] | None = None,
        dropout: tuple | str | float | None = None,
        norm_name: tuple | str = ("INSTANCE", {"affine": True}),
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
        language_model = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.trans_bias = trans_bias
        if filters is not None:
            self.filters = filters
            self.check_filters()
        else:
            self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num
        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: list[torch.Tensor] = [torch.rand(1)] * self.deep_supr_num
        if self.deep_supervision:
            self.deep_supervision_heads = self.get_deep_supervision_heads()
            self.check_deep_supr_num()

        self.apply(self.initialize_weights)
        self.check_kernel_stride()
        self.lang_encoder = language_model

        def create_skips(index, downsamples, upsamples, bottleneck, superheads=None):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """
            print(f"index: {index}")
            if len(downsamples) != len(upsamples):
                raise ValueError(f"{len(downsamples)} != {len(upsamples)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                print("at bottleneck")
                print(f"bottleneck type: {type(bottleneck)}")
                return bottleneck

            if superheads is None:
                print("no superheads")
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck)
                return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)

            super_head_flag = False
            if index == 0:  # don't associate a supervision head with self.input_block
                rest_heads = superheads
            else:
                if len(superheads) > 0:
                    super_head_flag = True
                    rest_heads = superheads[1:]
                else:
                    rest_heads = nn.ModuleList()

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, superheads=rest_heads)
            if super_head_flag:
                print("super head flag")
                return DynUNetSkipLayer(
                    index,
                    downsample=downsamples[0],
                    upsample=upsamples[0],
                    next_layer=next_layer,
                    heads=self.heads,
                    super_head=superheads[0],
                )

            return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)
        
        if not self.deep_supervision:
            self.skip_layers = create_skips(
                0, [self.input_block] + list(self.downsamples), self.upsamples[::-1], self.bottleneck
            )
        else:
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1],
                self.bottleneck,
                superheads=self.deep_supervision_heads,
            )

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if len(kernels) != len(strides) or len(kernels) < 3:
            raise ValueError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = f"length of kernel_size in block {idx} should be the same as spatial_dims."
                if len(kernel) != self.spatial_dims:
                    raise ValueError(error_msg)
            if not isinstance(stride, int):
                error_msg = f"length of stride in block {idx} should be the same as spatial_dims."
                if len(stride) != self.spatial_dims:
                    raise ValueError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise ValueError("deep_supr_num should be less than the number of up sample layers.")
        if deep_supr_num < 1:
            raise ValueError("deep_supr_num should be larger than 0.")

    def check_filters(self):
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        else:
            self.filters = filters[: len(self.strides)]
    
    def forward(self, x, ids, mask, token_type_ids):
        #print("found the model!")
        # Count the number of parameters
        #num_parameters = sum(p.numel() for p in self.lang_encoder.parameters())
        #print(x)
        #print("inside other forwards")
        #print(f"x size: {x.size()}")        


        lang_output = self.lang_encoder(ids, mask, token_type_ids)
        word_rep = lang_output[0]
        report_rep = lang_output[1]
        lang_rep = word_rep

        #print(f"output size: {lang_output.size()}")
        #print(lang_output)
        # Print the number of parameters
        #print(f"Number of Parameters in RobertaModel: {num_parameters}")

        #print(f"test: {type(self.skip_layers)}")
        #print(type(self.skip_layers))
        out = self.skip_layers(x, lang_rep)
        #print(f"out size: {out.size()}")

        out = self.output_block(out)
        #print(f"out size: {out.size()}")

        if self.training and self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(interpolate(feature_map, out.shape[2:]))
            print("use fancy return")
            return torch.stack(out_all, dim=1)

        del lang_output
        del word_rep
        del report_rep
        del lang_rep
        return out

    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=self.dropout)

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size, trans_bias=self.trans_bias
        )

    def get_module_list(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        kernel_size: Sequence[Sequence[int] | int],
        strides: Sequence[Sequence[int] | int],
        conv_block: type[nn.Module],
        upsample_kernel_size: Sequence[Sequence[int] | int] | None = None,
        trans_bias: bool = False,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(self.deep_supr_num)])

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


DynUnet = Dynunet = DynUNet
