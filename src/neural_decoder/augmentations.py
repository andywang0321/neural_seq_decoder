import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class LogZScore(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        # Ensure 1D (C,) tensors
        mean = mean.view(-1).float()
        std = std.view(-1).float()

        self.register_buffer("mean", mean)  # (C,)
        self.register_buffer("std", std)    # (C,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(torch.clamp(x, min=0.0))
        return (x - self.mean) / (self.std + self.eps)
    
class TimeMasking(nn.Module):
    """
    Randomly masks contiguous time steps in each sequence.
    Expects input of shape (B, T, F).

    Args:
        max_width: maximum width (in time steps) of each mask.
        n_masks: number of masked segments per sequence.
        p: probability of applying this augmentation at all.
        mask_value: what to put in the masked region.
    """
    def __init__(self, max_width: float, n_masks: int = 1, p: float = 0.5, mask_value: float = 0.0):
        super().__init__()
        self.max_width = max_width
        self.n_masks = n_masks
        self.p = p
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        if not self.training or torch.rand(()) > self.p or self.max_width <= 0 or self.n_masks <= 0:
            return x

        if x.dim() != 3:
            raise ValueError(f"TimeMasking expects (B, T, F), got {x.shape}")

        B, T, F = x.shape
        out = x.clone()

        for b in range(B):
            for _ in range(self.n_masks):
                width = torch.randint(1, int(self.max_width * T) + 1, (1,)).item()
                width = min(width, T)  
                start = torch.randint(0, T - width + 1, (1,)).item()
                out[b, start:start + width, :] = self.mask_value

        return out
    
class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _,_, C = x.shape
        noise = torch.randn(1, 1, C, device=x.device) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")
