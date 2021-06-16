"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from common.complex_modules import ComplexConv2d, ComplexConvTranspose2d, ComplexInstanceNorm2d, ComplexDropout2d, \
    complex_avg_pool2d
from data import transforms


def complex_to_chan_dim(x):
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

def chan_complex_to_last_dim(x):
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

class MultiDomainBlock(nn.Module):
    """
    A Convolutional-based Block that uses complex-valued features from both image and k-space domains.
    The computations in each domain consist of a convolution layer followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super(MultiDomainBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans*2, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width, 2]
        """
        image = complex_to_chan_dim(input)
        kspace = complex_to_chan_dim(transforms.fft2(input))
        image_output = chan_complex_to_last_dim(self.layers(image))
        kspace_output = chan_complex_to_last_dim(self.layers(kspace))
        kspace_output = transforms.ifft2(kspace_output)
        output = torch.cat((image_output, kspace_output), dim=1)
        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two MultiDomainBlocks
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.block1 = MultiDomainBlock(in_chans, out_chans, drop_prob)
        self.block2 = MultiDomainBlock(out_chans, out_chans, drop_prob)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width, 2]
        """
        return self.block2(self.block1(input))

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
               f'drop_prob={self.drop_prob})'


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional based Block that uses complex-valued features from both image and k-space domains.
    The computations in each domain consist of one convolution transpose layer followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super(TransposeConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans*2, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width, 2]
        """
        image = complex_to_chan_dim(input)
        kspace = complex_to_chan_dim(transforms.fft2(input))
        image_output = chan_complex_to_last_dim(self.layers(image))
        kspace_output = chan_complex_to_last_dim(self.layers(kspace))
        kspace_output = transforms.ifft2(kspace_output)
        output = torch.cat((image_output, kspace_output), dim=1)
        return output

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class MultiDomainModelV2(nn.Module):
    """
    PyTorch implementation of a multidomain U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super(MultiDomainModelV2, self).__init__()
        print('in multi-domain-model.')

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                ComplexConv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )]

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width, 2]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = complex_avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        output = transforms.root_sum_of_squares_complex(output, dim=1)

        return output
