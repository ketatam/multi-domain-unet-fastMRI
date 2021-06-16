import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """
    Applies a 2D convolution over a complex-valued input signal composed of several input planes using real numbers operations.
    Equivalent to performing standard 2D convolution with complex kernels and complex input.
    """
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        """
        Same arguments as torch.nn.Conv2d. See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super(ComplexConv2d, self).__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channel, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, out_channel, height, width, 2]
        """
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    """
    Applies a 2D transposed convolution operator over a complex-valued input image composed of several input planes using real numbers operations.
    Equivalent to performing standard 2D transposed convolution with complex kernels and complex input.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        """
        Same arguments as torch.nn.ConvTranspose2d. See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        """
        super(ComplexConvTranspose2d, self).__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channel, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, out_channel, height, width, 2]
        """
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexInstanceNorm2d(nn.Module):
    """
    Applies standard 2D Instance Normalization seperately on the real and imaginary part of the complex-valued input signal.
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, **kwargs):
        """
        Same arguments as torch.nn.InstanceNorm2d. See https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
        """
        super(ComplexInstanceNorm2d, self).__init__()
        self.in_re = nn.InstanceNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.in_im = nn.InstanceNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channel, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, out_channel, height, width, 2]
        """
        real = self.in_re(x[..., 0])
        imag = self.in_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class ComplexDropout2d(nn.Module):
    """
    Randomly zero out entire channels of the complex-valued input signal. (A channel contain the real and imoginary
    parts of the 2D input signal).
    """
    def __init__(self, p=0.5, inplace=False):
        """
        Same arguments as torch.nn.Dropout2d. See https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html
        """
        super(ComplexDropout2d, self).__init__()
        self.dropout = nn.Dropout2d(p=p, inplace=inplace)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channel, height, width, 2]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, out_channel, height, width, 2]
        """
        drop_mask = self.dropout(torch.ones_like(x[..., 0]))
        real = torch.mul(drop_mask, x[..., 0])
        imag = torch.mul(drop_mask, x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


def complex_avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                       divisor_override=None):
    """
    Applies 2D average-pooling operation seperately on the real and imaginary parts of a complex-valued input signal.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, in_channel, height, width, 2]
        The other arguments are the same as torch.nn.functional.avg_pool2d.
        See https://pytorch.org/docs/stable/nn.functional.html?highlight=avg_pool2d#torch.nn.functional.avg_pool2d
    Returns:
        (torch.Tensor): Output tensor of shape [batch_size, out_channel, height, width, 2]
    """
    real = F.avg_pool2d(input[..., 0],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                        divisor_override=divisor_override)
    imag = F.avg_pool2d(input[..., 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                        divisor_override=divisor_override)
    output = torch.stack((real, imag), dim=-1)
    return output
