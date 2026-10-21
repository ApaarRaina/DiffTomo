from difftomo.data import SheppLogan
from difftomo.CT.Projector import ParallelBeamProjector, FanBeamProjector
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import pytest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

@pytest.mark.parametrize("projections", [1000, 500, 250, 100])
@pytest.mark.parametrize("detectors", [900, 800, 500, 200])
@pytest.mark.parametrize("device", ["cuda"])
def test_parallelbeamprojector(projections, detectors, device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(projections=projections, detector_bins=detectors, device=device)
    img = phantom.data.to(device)
    img_batch = torch.stack([img.unsqueeze(0), img.unsqueeze(0), img.unsqueeze(0)])
    sinogram = projector.forward(img)
    sinogram_batch = projector.forward(img_batch)

    projector_full = ParallelBeamProjector(projections=1000, detector_bins=900, device=device)
    sinogram_full = projector_full.forward(img)
    sinogram_full_batch = projector_full.forward(img_batch)
    fbp_full = projector_full.FBP(sinogram_full)
    fbp_full_batch = projector_full.FBP(sinogram_full_batch)

    psnr = peak_signal_noise_ratio(img.unsqueeze(0).unsqueeze(0).cpu().numpy(), fbp_full.cpu().numpy())

    assert sinogram.shape == (1, 1, projections, detectors), "Sinogram shape is incorrect"
    assert isinstance(sinogram, torch.Tensor), "Sinogram is not a torch tensor"
    assert sinogram.device.type == device , "Sinogram is not on the correct device"
    assert fbp_full.shape == img.unsqueeze(0).unsqueeze(0).shape, "FBP shape is incorrect"
    assert psnr > 25, "PSNR is too low"
    assert sinogram_batch.shape == (3, 1, projections, detectors), "Sinogram shape is incorrect"
    assert isinstance(sinogram_batch, torch.Tensor), "Sinogram batch is not a torch tensor"
    assert fbp_full_batch.shape == img_batch.shape, "FBP batch shape is incorrect"
    assert isinstance(fbp_full, torch.Tensor), "FBP is not a torch tensor"


@pytest.mark.parametrize("projections", [1000, 500, 250, 100])
@pytest.mark.parametrize("detectors", [900, 800, 500, 200])
@pytest.mark.parametrize("device", ["cuda"])
def test_fanbeamprojector(projections, detectors, device):
    phantom = SheppLogan()
    projector = FanBeamProjector(projections=projections, detector_bins=detectors, device=device)
    img = phantom.data.to(device)
    img_batch = torch.stack([img.unsqueeze(0), img.unsqueeze(0), img.unsqueeze(0)])
    sinogram = projector.forward(img)
    sinogram_batch = projector.forward(img_batch)

    projector_full = FanBeamProjector(projections=1000, detector_bins=900, device=device)
    sinogram_full = projector_full.forward(img)
    sinogram_full_batch = projector_full.forward(img_batch)
    fbp_full = projector_full.FBP(sinogram_full)
    fbp_full_batch = projector_full.FBP(sinogram_full_batch)

    psnr = peak_signal_noise_ratio(img.unsqueeze(0).unsqueeze(0).cpu().numpy(), fbp_full.cpu().numpy())

    print(psnr)
    assert sinogram.shape == (1, 1, projections, detectors), "Sinogram shape is incorrect"
    assert isinstance(sinogram, torch.Tensor), "Sinogram is not a torch tensor"
    assert sinogram.device.type == device , "Sinogram is not on the correct device"
    assert fbp_full.shape == img.unsqueeze(0).unsqueeze(0).shape, "FBP shape is incorrect"
    assert psnr > 25, "PSNR is too low"
    assert sinogram_batch.shape == (3, 1, projections, detectors), "Sinogram shape is incorrect"
    assert isinstance(sinogram_batch, torch.Tensor), "Sinogram batch is not a torch tensor"
    assert fbp_full_batch.shape == img_batch.shape, "FBP batch shape is incorrect"
    assert isinstance(fbp_full, torch.Tensor), "FBP is not a torch tensor"


# @pytest.mark.parametrize("projections", [1000, 100])
# @pytest.mark.parametrize("detectors", [900, 200])
# @pytest.mark.parametrize("device", ["cpu"])
# def test_fanbeamprojector_cpu(projections, detectors, device):
#     phantom = SheppLogan()
#     projector = FanBeamProjector(projections=projections, detector_bins=detectors, device=device)
#     img = phantom.data
#     img_batch = torch.stack([phantom.data.unsqueeze(0), phantom.data.unsqueeze(0), phantom.data.unsqueeze(0)])

#     sinogram = projector.forward(img)
#     sinogram_batch = projector.forward(img_batch)

#     projector_full = FanBeamProjector(projections=1000, detector_bins=800, device=device)
#     sinogram_full = projector_full.forward(img)
#     sinogram_full_batch = projector_full.forward(img_batch)
#     fbp_full = projector_full.FBP(sinogram_full)
#     fbp_full_batch = projector_full.FBP(sinogram_full_batch)

#     psnr = peak_signal_noise_ratio(img.cpu().numpy(), fbp_full.cpu().numpy())
    
#     assert sinogram.shape == (1, 1, projections, detectors), "Sinogram shape is incorrect"
#     assert isinstance(sinogram, torch.Tensor), "Sinogram is not a torch tensor"
#     assert sinogram.device == device , "Sinogram is not on the correct device"
#     assert fbp_full.shape == img.shape, "FBP shape is incorrect"
#     assert psnr > 30, "PSNR is too low"
#     assert sinogram_batch.shape == (3, 1, projections, detectors), "Sinogram shape is incorrect"
#     assert isinstance(sinogram_batch, torch.Tensor), "Sinogram batch is not a torch tensor"
#     assert fbp_full_batch.shape == img_batch.shape, "FBP batch shape is incorrect"
#     assert isinstance(fbp_full, torch.Tensor), "FBP is not a torch tensor"
    

# @pytest.mark.parametrize("projections", [1000, 100])
# @pytest.mark.parametrize("detectors", [900, 200])
# @pytest.mark.parametrize("device", ["cpu"])
# def test_parallelbeamprojector_cpu(projections, detectors, device):
#     phantom = SheppLogan()
#     projector = ParallelBeamProjector(projections=projections, detector_bins=detectors, device=device)
#     img = phantom.data
#     img_batch = torch.stack([phantom.data.unsqueeze(0), phantom.data.unsqueeze(0), phantom.data.unsqueeze(0)])
#     sinogram = projector.forward(img)
#     sinogram_batch = projector.forward(img_batch)

#     projector_full = ParallelBeamProjector(projections=1000, detector_bins=800, device=device)
#     sinogram_full = projector_full.forward(img)
#     sinogram_full_batch = projector_full.forward(img_batch)
#     fbp_full = projector_full.FBP(sinogram_full)
#     fbp_full_batch = projector_full.FBP(sinogram_full_batch)

#     psnr = peak_signal_noise_ratio(img.cpu().numpy(), fbp_full.cpu().numpy())
    
    
#     assert sinogram.shape == (1, 1, projections, detectors), "Sinogram shape is incorrect"
#     assert isinstance(sinogram, torch.Tensor), "Sinogram is not a torch tensor"
#     assert sinogram.device == device , "Sinogram is not on the correct device"
#     assert fbp_full.shape == img.shape, "FBP shape is incorrect"
#     assert psnr > 30, "PSNR is too low"
#     assert sinogram_batch.shape == (3, 1, projections, detectors), "Sinogram shape is incorrect"
#     assert isinstance(sinogram_batch, torch.Tensor), "Sinogram batch is not a torch tensor"
#     assert fbp_full_batch.shape == img_batch.shape, "FBP batch shape is incorrect"
#     assert isinstance(fbp_full, torch.Tensor), "FBP is not a torch tensor"