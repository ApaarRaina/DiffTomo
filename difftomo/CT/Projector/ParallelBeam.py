import torch
from ._extensions import forward_siddon, backward_siddon
import math


class _ParallelBeamFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image,
        projections,
        detectors,
        detector_spacing,
        coverage,
        distance,
    ):
        # CUDA A
        sinogram = forward_siddon(
            image,
            projections,
            detectors,
            detector_spacing,
            coverage,
            distance,
        )

        # Save what we need to perform A^T later
        ctx.image_shape = image.shape
        ctx.projections = projections
        ctx.detectors = detectors
        ctx.detector_spacing = detector_spacing
        ctx.coverage = coverage
        ctx.distance = distance

        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        # CUDA A^T
        grad_image = backward_siddon(
            grad_sinogram,
            ctx.image_shape,
            ctx.projections,
            ctx.detectors,
            ctx.detector_spacing,
            ctx.coverage,
            ctx.distance,
        )

        # Gradients corresponding to the inputs of forward()
        return (
            grad_image,  # image
            None,  # projections
            None,  # detectors
            None,  # detector_spacing
            None,  # coverage
            None,  # distance
        )

class _FBPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sinogram,
        image_shape,
        projections,
        detectors,
        detector_spacing,
        coverage,
        distance,
    ):
        # CUDA A^T
        recon_image = backward_siddon(
            sinogram,
            image_shape,
            projections,
            detectors,
            detector_spacing,
            coverage,
            distance,
        )

        # Save what we need to perform A^T later
        ctx.image_shape = image_shape
        ctx.projections = projections
        ctx.detectors = detectors
        ctx.detector_spacing = detector_spacing
        ctx.coverage = coverage
        ctx.distance = distance

        return recon_image

    @staticmethod
    def backward(ctx, grad_recon_image):
        # CUDA A
        grad_sinogram = forward_siddon(
            grad_recon_image.contiguous(),
            ctx.projections,
            ctx.detectors,
            ctx.detector_spacing,
            ctx.coverage,
            ctx.distance,
        )

        # Gradients corresponding to the inputs of forward()
        return (
            grad_sinogram,  # sinogram
            None,  # image_shape
            None,  # projections
            None,  # detectors
            None,  # detector_spacing
            None,  # coverage
            None,  # distance
            None,  # filter_type
        )


class ParallelBeamProjector:
    def __init__(
        self,
        projections=800,
        detector_bins=800,
        detector_spacing=1,
        coverage=2 * torch.pi,
        distance=300,
        type="line",
        device="cpu",
    ):
        self.projections = projections
        self.detectors = detector_bins
        self.detector_spacing = detector_spacing
        self.coverage = coverage
        self.distance = distance
        self.type = type
        self.device = device

    def forward(self, image):
        self.image_shape = image.shape

        if self.type == "line":
            return _ParallelBeamFunction.apply(
                image,
                self.projections,
                self.detectors,
                self.detector_spacing,
                self.coverage,
                self.distance,
            )

        raise ValueError(f"Unknown projector type: {self.type}")

    # function to perform the backward operation without autograd
    def backward(self, sinogram, image_shape=None):
        if image_shape is not None:
            self.image_shape = image_shape
        elif not hasattr(self, "image_shape"):
            raise Warning(
                "Image shape must be provided for backward operation. Will use the default image shape (512x512)."
            )
        if sinogram.shape[0] != self.projections or sinogram.shape[1] != self.detectors:
            raise ValueError(
                f"Sinogram shape {sinogram.shape} does not match projector configuration "
                f"({self.projections}, {self.detectors})"
            )

        if self.type == "line":
            return backward_siddon(
                sinogram,
                self.image_shape,
                self.projections,
                self.detectors,
                self.detector_spacing,
                self.coverage,
                self.distance,
            )

        raise ValueError(f"Unknown projector type: {self.type}")

    def _sinc(self, x):
        px = math.pi * x
        zero = x == 0
        safe = torch.where(zero, torch.ones_like(px), px)
        return torch.where(zero, torch.ones_like(x), torch.sin(px) / safe)

    def FBP(self, sinogram, image_shape=None, filter_type="ram-lak"):

        if image_shape is not None:
            self.image_shape = image_shape

        elif not hasattr(self, "image_shape"):
            raise Warning(
                "Image shape must be provided for backward operation. Will use the default image shape (512x512)."
            )
        
        if sinogram.shape[0] != self.projections or sinogram.shape[1] != self.detectors:
            raise ValueError(
                f"Sinogram shape {sinogram.shape} does not match projector configuration "
                f"({self.projections}, {self.detectors})"
            )

        if filter_type not in ["ram-lak", "shepp-logan", "cosine", "hamming", "hann"]:
            raise ValueError(
                f"Unknown filter type: {filter_type}. Must be one of ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']"
            )

        n = self.detectors
        pad_len = max(64, 2 ** int(torch.ceil(torch.log2(torch.tensor(2 * n)))))
        scale = torch.tensor(torch.pi / (self.projections)).to(self.device)

        freqs = torch.fft.fftfreq(pad_len, d=float(self.detector_spacing)).to(sinogram.device)
        filt = torch.abs(freqs)
        f_max = 1.0 / (2.0 * float(self.detector_spacing))
        x = filt / f_max

        if filter_type == "shepp-logan":
            filt = filt * self._sinc(x / 2)            
        elif filter_type == "cosine":
            filt = filt * torch.cos(torch.pi * x / 2)
        elif filter_type == "hamming":
            filt = filt * (0.54 + 0.46 * torch.cos(torch.pi * x))
        elif filter_type == "hann":
            filt = filt * 0.5 * (1 + torch.cos(torch.pi * x))
    
        if self.type == "line":
            sino_padded = torch.nn.functional.pad(sinogram, (0, pad_len - n))
            F_Rf = torch.fft.fft(sino_padded, dim=1).to(self.device)
            F_Rf_filtered = F_Rf * filt[None, :].to(self.device)
            Rf_filtered = torch.fft.ifft(F_Rf_filtered, dim=1).real[: , :n].contiguous().to(self.device)

            return _FBPFunction.apply(
                Rf_filtered,
                self.image_shape,
                self.projections,
                self.detectors,
                self.detector_spacing,
                self.coverage,
                self.distance,
            ) * scale

        raise ValueError(f"Unknown projector type: {self.type}")
