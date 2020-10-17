import torch
from typing import Tuple

def fft(data: torch.Tensor, dim: Tuple[int, ...] = None) -> torch.Tensor:
    """ 
    Computes the Fourier transform from (image to k-space)

    :param data: image data as complex torch.Tensor
    :param dim: dimensions to transform
    :returns: transformed data
    """

    if not dim:
        dim = tuple([x for x in range(data.ndim)])
        # Since torch can only do up to 3 dimensions, we will pick the last 3
        if len(dim) > 3:
            dim = dim[-3:]
    
    # Torch ffts only support 1, 2, or 3 dimensions
    assert len(dim) <= 3 and len(dim) >= 1

    # ifftshift
    shifts = [(data.shape[x] + 1) // 2 for x in dim]
    data = torch.roll(data, shifts, dim)

    # Permute, ifft, unpermute
    permute_dim,unpermute_dim = get_fft_permute_dims(data.ndim, dim)
    data = data.permute(permute_dim)
    data = torch.view_as_complex(torch.fft(torch.view_as_real(data), len(dim)))
    data = data.permute(unpermute_dim)

    # fftshift
    shifts = [(data.shape[x]) // 2 for x in dim]
    data = torch.roll(data, shifts, dim)

    # Scaling
    data /= torch.sqrt(torch.prod(torch.Tensor([data.shape[d] for d in dim])))

    return data

def ifft(data: torch.Tensor, dim: Tuple[int, ...] = None) -> torch.Tensor:
    """ 
    Computes the inverse Fourier transform from (k-space to image)

    :param data: k-space data as complex torch.Tensor
    :param dim: dimensions to transform
    :returns: transformed data
    """

    if not dim:
        dim = tuple([x in range(data.ndim)])
        # Since torch can only do up to 3 dimensions, we will pick the last 3
        if len(dim) > 3:
            dim = dim[-3:]
    
    # Torch ffts only support 1, 2, or 3 dimensions
    assert len(dim) <= 3 and len(dim) >= 1

    # ifftshift
    shifts = [(data.shape[x] + 1) // 2 for x in dim]
    data = torch.roll(data, shifts, dim)

    # Permute, ifft, unpermute
    permute_dim,unpermute_dim = get_fft_permute_dims(data.ndim, dim)
    data = data.permute(permute_dim)
    data = torch.view_as_complex(torch.ifft(torch.view_as_real(data), len(dim)))
    data = data.permute(unpermute_dim)

    # fftshift
    shifts = [(data.shape[x]) // 2 for x in dim]
    data = torch.roll(data, shifts, dim)

    # Scaling
    data *= torch.sqrt(torch.prod(torch.Tensor([data.shape[d] for d in dim])))

    return data

def get_fft_permute_dims(ndim: int, transform_dim: Tuple[int,...]) -> Tuple[Tuple[int,...],Tuple[int,...]]:
    """
    Helper function for determiming permute dimensions for FFT/IFFT.
    
    Returns permutation orders needed for moving the transform dimensions `transform_dim` to the highest dimensions and back.

    :param ndim: total number of dimensions
    :param transform_dim: dimensions to transform
    :return permute_dim,unpermute_dim: tuples needed to permute and unpermute

    """
    dd = [d%ndim for d in transform_dim]
    permute_dims = []
    for d in range(ndim):
        if d not in dd:
            permute_dims.append(d)
    for d in dd:
        permute_dims.append(d%ndim)

    permute_dims = tuple(permute_dims)

    unpermute_dims = [0 for _ in range(ndim)]
    for i, d in enumerate(permute_dims):
        unpermute_dims[d] = i
    unpermute_dims = tuple(unpermute_dims)

    return permute_dims,unpermute_dims