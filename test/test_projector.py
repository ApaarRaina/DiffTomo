from difftomo.data import SheppLogan
from difftomo.CT.Projector import ParallelBeamProjector
from difftomo.CT.Iterative import TVDenoising, SIRT, TGVDenoising
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import pytest
import torch
import torch.nn as nn

@pytest.mark.parametrize("projections", [1000, 500, 250, 100])
@pytest.mark.parametrize("detectors", [900, 800, 500, 200])
@pytest.mark.parametrize("device", ["cuda"])
def test_parallelbeamprojector(projections, detectors, device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(projections=projections, detector_bins=detectors, device=device)
    img = phantom.data
    sinogram = projector.forward(img)

    projector_full = ParallelBeamProjector(projections=1000, detector_bins=800, device=device)
    sinogram_full = projector_full.forward(img)
    fbp_full = projector_full.FBP(sinogram_full)

    psnr = peak_signal_noise_ratio(img.cpu().numpy(), fbp_full.cpu().numpy())
    
    
    assert sinogram.shape == (projections, detectors), "Sinogram shape is incorrect"
    assert isinstance(sinogram, torch.Tensor), "Sinogram is not a torch tensor"
    assert psnr > 30, "PSNR is too low"
    assert isinstance(fbp_full, torch.Tensor), "FBP is not a torch tensor"