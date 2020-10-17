from InnerEye.Reconstruction.recon_utils import fft
import torch

def test_permute_util() -> None:
    assert fft.get_fft_permute_dims(5,(0,1)) == ((2,3,4,0,1),(3,4,0,1,2))
    assert fft.get_fft_permute_dims(5,(-2,-1)) == ((0,1,2,3,4),(0,1,2,3,4))

def test_fft_noise_scaling() -> None:
    noise = torch.randn(256,256, dtype=torch.cfloat)
    assert (torch.abs(torch.std(noise) - 1.0) < 0.01)

    # Noise std in k-space should also be 1.0 
    knoise = fft.fft(noise)
    assert (torch.abs(torch.std(knoise) - 1.0) < 0.01)

    # Noise after recon should also be 1.0 
    rnoise = fft.fft(knoise)
    assert (torch.abs(torch.std(rnoise) - 1.0) < 0.01)