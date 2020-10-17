import torch

def ifft(data: torch.Tensor, dim: Tuple[int, ...]=None):
    """ 

    Computes the inverse Fourier transform from (k-space to image)

    :param data: k-space data as complex torch.Tensor
    :param dim: vector of dimensions to transform
    :returns: transformed data

    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

# Cell
def fft2(data):
    """
    Fourier transform tensor along dimensions -2 and -3.

    data is a torch.Tensor where the last dimention is of length 2 (real, imag).
    """
    assert data.size(-1) == 2
    data = torch.roll(data, shifts=((data.shape[-3] + 1) // 2, (data.shape[-2] + 1) // 2), dims=(-3,-2))
    data = torch.fft(data, 2)
    data = torch.roll(data, shifts=(data.shape[-3] // 2, data.shape[-2] // 2), dims=(-3,-2))
    scale = torch.sqrt(torch.Tensor([1.0*data.shape[-2]*data.shape[-3]]).to(data.device))
    data /= scale
    return data

# Cell
def ifft2(data):
    """
    Fourier transform tensor along dimensions -2 and -3.

    data is a torch.Tensor where the last dimention is of length 2 (real, imag).
    """
    assert data.size(-1) == 2
    data = torch.roll(data, shifts=((data.shape[-3] + 1) // 2, (data.shape[-2] + 1) // 2), dims=(-3,-2))
    data = torch.ifft(data, 2)
    data = torch.roll(data, shifts=(data.shape[-3] // 2, data.shape[-2] // 2), dims=(-3,-2))
    scale = torch.sqrt(torch.Tensor([1.0*data.shape[-2]*data.shape[-3]]).to(data.device))
    data *= scale
    return data