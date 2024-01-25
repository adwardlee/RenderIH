from torchvision.transforms.functional import *


def denormalize(tensor, mean, std, inplace=False):
    if not torch.is_tensor(tensor) or tensor.ndimension() != 3:
        raise TypeError("invalid tensor or tensor channel is not CHW")

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).sub_(-1 * mean[:, None, None])
    return tensor


def batch_denormalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError("invalid tensor or tensor channel is not BCHW")

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[None, :, None, None]).sub_(-1 * mean[None, :, None, None])
    return tensor


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    else:
        return tensor


def bhwc_2_bchw(tensor):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError("invalid tensor or tensor channel is not BCHW")
    return tensor.unsqueeze(1).transpose(1, -1).squeeze(-1)


def bchw_2_bhwc(tensor):
    """
    :param x: torch tensor, B x C x H x W
    :return:  torch tensor, B x H x W x C
    """
    if not torch.is_tensor(tensor) or tensor.ndimension() != 4:
        raise TypeError("invalid tensor or tensor channel is not BCHW")
    return tensor.unsqueeze(-1).transpose(1, -1).squeeze(1)
