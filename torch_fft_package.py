import numpy as np
import torch.fft


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension containing
            real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    #     assert data.size(-1) == 2
    #     ndims = len(list(data.size()))

    #     if ndims == 5:
    #         data = data.permute(0,3,1,2,4)
    #     elif ndims == 6:
    #         data = data.permute(0,3,4,1,2,5)
    #     else:
    #         raise ValueError('fft2: ndims > 6 not supported!')

    data = ifftshift(data, dim=(-2, -1))
    data = torch.fft.fft2(data, dim=(-2, -1), norm="ortho")
    data = fftshift(data, dim=(-2, -1))

    #     if ndims == 5:
    #         data = data.permute(0,2,3,1,4)
    #     elif ndims == 6:
    #         data = data.permute(0,3,4,1,2,5)
    #     else:
    #         raise ValueError('fft2: ndims > 6 not supported!')

    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension containing
            real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.

    Returns:
        torch.Tensor: The IFFT of the input.
    """

    #     if ndims == 5:
    #         data = data.permute(0,3,1,2,4)
    #     elif ndims == 6:
    #         data = data.permute(0,3,4,1,2,5)
    #     else:
    #         raise ValueError('ifft2: ndims > 6 not supported!')

    data = ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifft2(data, dim=(-2, -1), norm="ortho")
    data = fftshift(data, dim=(-2, -1))

    #     if ndims == 5:
    #         data = data.permute(0,2,3,1,4)
    #     elif ndims == 6:
    #         data = data.permute(0,3,4,1,2,5)
    #     else:
    #         raise ValueError('ifft2: ndims > 6 not supported!')

    return data


def create_center_mask(undersample_ratio=3.3, shape=(96, 96)):
    # Create a binary mask for central k-space lines
    center_fraction = 1 / undersample_ratio
    _, height = shape
    mask = np.zeros(shape, dtype=np.bool_)
    center = height // 2
    offset = int(np.floor(center_fraction * height / 2))
    mask[:, center-offset:center+offset] = True
    return torch.from_numpy(mask)
