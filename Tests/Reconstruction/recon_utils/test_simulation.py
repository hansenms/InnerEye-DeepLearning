from InnerEye.Reconstruction.recon_utils import simulation

import numpy as np

def test_generate_birdcage_sensitivities() -> None:
    coils = simulation.generate_birdcage_sensitivities(matrix_size=128, number_of_coils=4)
    assert coils.shape == (4,128,128)
    assert coils.dtype == np.dtype('complex64')
