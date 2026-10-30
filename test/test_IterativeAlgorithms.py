from difftomo.data import SheppLogan
from difftomo.CT.Projector import ParallelBeamProjector
from difftomo.CT.Iterative import TVDenoising, SIRT, TGVDenoising, ART, SART, CGLS
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn


def sobel_filter(image, device="cpu"):
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
    ).to(device)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
    ).to(device)

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    grad_x = nn.functional.conv2d(image, sobel_x, padding=1)
    grad_y = nn.functional.conv2d(image, sobel_y, padding=1)

    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


@pytest.mark.parametrize("device", ["cuda"])
def test_SIRT(device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(device=device)
    sirt_10 = SIRT(projector, iterations=10, device=device)
    sirt_25 = SIRT(projector, iterations=25, device=device)
    sirt_50 = SIRT(projector, iterations=50, device=device)

    original_img = phantom.data.to(device)
    sinogram = projector.forward(original_img)
    sinogram = sinogram.to(device)

    backprojected_img = projector.backward(sinogram, image_shape=(400, 400))
    reconstructed_10, reconstructed_25, reconstructed_50 = (
        sirt_10.reconstruct(sinogram, image_shape=(400, 400)),
        sirt_25.reconstruct(sinogram, image_shape=(400, 400)),
        sirt_50.reconstruct(sinogram, image_shape=(400, 400)),
    )

    original_img_np = original_img.cpu().numpy()
    data_range = np.max(original_img_np) - np.min(original_img_np)

    psnr1 = peak_signal_noise_ratio(
        original_img_np,
        backprojected_img.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr2 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_10.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr3 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_25.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr4 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_50.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )

    rmse1, rmse2, rmse3, rmse4 = (
        rmse(original_img_np, backprojected_img.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_10.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_25.squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_50.squeeze(0).cpu().numpy()),
    )

    assert psnr1 < psnr2 < psnr3 < psnr4
    assert rmse1 > rmse2 > rmse3 > rmse4


@pytest.mark.parametrize("device", ["cpu"])
def test_ART(device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(projections=300, detector_bins=500, device=device)
    ART_1 = ART(projector, iterations=1)
    ART_5 = ART(projector, iterations=5)
    ART_10 = ART(projector, iterations=10)

    original_img = phantom.data.to(device)
    sinogram = projector.forward(original_img)
    sinogram = sinogram.to(device)

    backprojected_img = projector.backward(sinogram, image_shape=(400, 400))
    reconstructed_1, reconstructed_5, reconstructed_10 = (
        ART_1.reconstruct(sinogram, image_shape=(400, 400)),
        ART_5.reconstruct(sinogram, image_shape=(400, 400)),
        ART_10.reconstruct(sinogram, image_shape=(400, 400)),
    )

    original_img_np = original_img.cpu().numpy()
    data_range = np.max(original_img_np) - np.min(original_img_np)

    psnr1 = peak_signal_noise_ratio(
        original_img_np,
        backprojected_img.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr2 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_1.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr3 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_5.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr4 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_10.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )

    rmse1, rmse2, rmse3, rmse4 = (
        rmse(original_img_np, backprojected_img.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_1.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_5.squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_10.squeeze(0).cpu().numpy()),
    )

    assert psnr1 < psnr2 < psnr3 < psnr4
    assert rmse1 > rmse2 > rmse3 > rmse4


@pytest.mark.parametrize("device", ["cuda"])
def test_SART(device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(device=device)
    SART_1 = SART(projector, iterations=1, device=device)
    SART_5 = SART(projector, iterations=5, device=device)
    SART_10 = SART(projector, iterations=10, device=device)

    original_img = phantom.data.to(device)
    sinogram = projector.forward(original_img)
    sinogram = sinogram.to(device)

    backprojected_img = projector.backward(sinogram, image_shape=(400, 400))
    reconstructed_1, reconstructed_5, reconstructed_10 = (
        SART_1.reconstruct(sinogram, image_shape=(400, 400)),
        SART_5.reconstruct(sinogram, image_shape=(400, 400)),
        SART_10.reconstruct(sinogram, image_shape=(400, 400)),
    )

    original_img_np = original_img.cpu().numpy()
    data_range = np.max(original_img_np) - np.min(original_img_np)

    psnr1 = peak_signal_noise_ratio(
        original_img_np,
        backprojected_img.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr2 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_1.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr3 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_5.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr4 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_10.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )

    rmse1, rmse2, rmse3, rmse4 = (
        rmse(original_img_np, backprojected_img.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_1.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_5.squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_10.squeeze(0).cpu().numpy()),
    )

    assert psnr1 < psnr2 < psnr3 < psnr4
    assert rmse1 > rmse2 > rmse3 > rmse4


@pytest.mark.parametrize("device", ["cuda"])
def test_CGLS(device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(device=device)
    CGLS_1 = CGLS(projector, iterations=1, device=device)
    CGLS_5 = CGLS(projector, iterations=5, device=device)
    CGLS_10 = CGLS(projector, iterations=10, device=device)

    original_img = phantom.data.to(device)
    sinogram = projector.forward(original_img)
    sinogram = sinogram.to(device)

    backprojected_img = projector.backward(sinogram, image_shape=(400, 400))
    reconstructed_1, reconstructed_5, reconstructed_10 = (
        CGLS_1.reconstruct(sinogram, image_shape=(400, 400)),
        CGLS_5.reconstruct(sinogram, image_shape=(400, 400)),
        CGLS_10.reconstruct(sinogram, image_shape=(400, 400)),
    )

    original_img_np = original_img.cpu().numpy()
    data_range = np.max(original_img_np) - np.min(original_img_np)

    psnr1 = peak_signal_noise_ratio(
        original_img_np,
        backprojected_img.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr2 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_1.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr3 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_5.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )
    psnr4 = peak_signal_noise_ratio(
        original_img_np,
        reconstructed_10.squeeze(0).squeeze(0).cpu().numpy(),
        data_range=data_range,
    )

    rmse1, rmse2, rmse3, rmse4 = (
        rmse(original_img_np, backprojected_img.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_1.squeeze(0).squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_5.squeeze(0).cpu().numpy()),
        rmse(original_img_np, reconstructed_10.squeeze(0).cpu().numpy()),
    )

    assert psnr1 < psnr2 < psnr3 < psnr4
    assert rmse1 > rmse2 > rmse3 > rmse4


@pytest.mark.parametrize("device", ["cuda"])
def test_TV(device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(projections=100, device=device)
    TV_10 = TVDenoising(projector, iterations=10, lamda=0.05, device=device)
    TV_15 = TVDenoising(projector, iterations=15, lamda=0.05, device=device)
    TV_25 = TVDenoising(projector, iterations=25, lamda=0.05, device=device)

    original_img = phantom.data.to(device)
    sinogram = projector.forward(original_img)
    fbp_sparse = projector.FBP(sinogram)

    reconstructed_10, reconstructed_15, reconstructed_25 = (
        TV_10.reconstruct(fbp_sparse),
        TV_15.reconstruct(fbp_sparse),
        TV_25.reconstruct(fbp_sparse),
    )

    sobel_mag_10, sobel_mag_15, sobel_mag_25 = (
        sobel_filter(reconstructed_10, reconstructed_10.device),
        sobel_filter(reconstructed_15, reconstructed_15.device),
        sobel_filter(reconstructed_25, reconstructed_25.device),
    )

    assert torch.sum(sobel_mag_10) > torch.sum(sobel_mag_15) > torch.sum(sobel_mag_25)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("iterations", [10, 15, 25])
def test_TGV(iterations, device):
    phantom = SheppLogan()
    projector = ParallelBeamProjector(projections=100, device=device)
    TV = TVDenoising(projector, iterations=iterations, lamda=0.05, device=device)
    TGV = TGVDenoising(projector, iterations=iterations, lamda=[1, 2], device=device)

    original_img = phantom.data.to(device)
    sinogram = projector.forward(original_img)
    fbp_sparse = projector.FBP(sinogram)

    TV_recon, TGV_recon = TV.reconstruct(fbp_sparse), TGV.reconstruct(fbp_sparse)

    sobel_mag_TV = sobel_filter(TV_recon, TV_recon.device)
    sobel_mag_TGV = sobel_filter(TGV_recon, TGV_recon.device)

    assert torch.sum(sobel_mag_TV) < torch.sum(sobel_mag_TGV)
