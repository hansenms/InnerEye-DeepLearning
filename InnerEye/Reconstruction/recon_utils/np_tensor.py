import numpy as np
import torch

def as_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Takes a real or complex numpy array and returns a stacked compled tensor

    :param data: numpy array
    :return PyTorch representation of the numpy array
    """
    real = np.real(data).copy()
    imag = np.imag(data).copy()
    return torch.stack([torch.Tensor(real), torch.Tensor(imag)], dim=data.ndim)

def as_numpy(data: torch.Tensor) -> np.ndarray:
    """
    Takes a complex (or real) pytorch tensor and converts to complex (or real) numpy

    param: data: PyTorch tensor
    return: numpy representation of the PyTorch tensor
    """
    if data.shape[-1] == 2:
        return np.array(data[...,-2]) + 1j*np.array(data[...,-1])
    
    if data.shape[-1] == 1:
        data = data.squeeze(-1) 
        
    return data.numpy()