from InnerEye.Reconstruction.recon_utils import np_tensor

import numpy as np
import torch

def test_to_tensor_cplx() -> None:
    cplx_arr = np.random.rand(128,128) + 1j*np.random.rand(128,128)
    cplx_ten = np_tensor.as_tensor(cplx_arr)
    assert cplx_ten.shape == (128,128,2)
    assert np.sum(cplx_ten[...,-2].numpy() - np.real(cplx_arr)) < 1e-5
    assert np.sum(cplx_ten[...,-1].numpy() - np.imag(cplx_arr)) < 1e-5

def test_to_tensor_real() -> None:
    real_arr = np.random.rand(128,128)
    real_ten = np_tensor.as_tensor(real_arr)
    assert real_ten.shape == (128,128,2)
    assert np.sum(real_ten[...,-2].numpy() - np.real(real_arr)) < 1e-5
    assert np.sum(real_ten[...,-1].numpy()) < 1e-10

def test_to_numpy_cplx() -> None:
    cplx_ten = torch.rand((128,128,2), dtype=torch.float32)
    cplx_arr = np_tensor.as_numpy(cplx_ten)
    assert cplx_arr.shape == (128,128)
    assert np.sum(cplx_ten[...,-2].numpy() - np.real(cplx_arr)) < 1e-5
    assert np.sum(cplx_ten[...,-1].numpy() - np.imag(cplx_arr)) < 1e-5

def test_to_numpy_cplx() -> None:
    cplx_ten = torch.rand((128,128,2), dtype=torch.float32)
    cplx_arr = np_tensor.as_numpy(cplx_ten)
    assert cplx_arr.shape == (128,128)
    assert np.sum(cplx_ten[...,-2].numpy() - np.real(cplx_arr)) < 1e-5
    assert np.sum(cplx_ten[...,-1].numpy() - np.imag(cplx_arr)) < 1e-5

def test_to_numpy_real() -> None:
    real_ten = torch.rand((128,128,1), dtype=torch.float32)
    real_arr = np_tensor.as_numpy(real_ten)
    assert real_arr.shape == (128,128)
    assert np.sum(real_ten[...,0].numpy() - np.real(real_arr)) < 1e-5
    assert np.sum(np.imag(real_arr)) < 1e-10

    real_ten = torch.rand((128,128), dtype=torch.float32)
    real_arr = np_tensor.as_numpy(real_ten)
    assert real_arr.shape == (128,128)
    assert np.sum(real_ten[...,0].numpy() - np.real(real_arr)) < 1e-5
    assert np.sum(np.imag(real_arr)) < 1e-10