import torch
from ._extensions import forward_siddon, backward_siddon, forward_siddon_cpu, backward_siddon_cpu
import math
import warnings

class _FanBeamFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image,
        projections,
        detectors,
        detector_spacing,
        coverage,
        distance,
        workers=16,
        device='cuda',
    ):
        ctx.type = 'f'
        ctx.device = device
        ctx.workers = workers
        # CUDA A
        if ctx.device == 'cuda':
            sinogram = forward_siddon(
                image,
                projections,
                detectors,
                detector_spacing,
                coverage,
                distance,
                ctx.type
            )
        else:
            sinogram = forward_siddon_cpu(
                image,
                projections,
                detectors,
                detector_spacing,
                coverage,
                distance,
                ctx.type,
                ctx.workers
            )

        # Save what we need to perform A^T later
        ctx.image_shape = image.shape
        ctx.projections = projections
        ctx.detectors = detectors
        ctx.detector_spacing = detector_spacing
        ctx.coverage = coverage
        ctx.distance = distance

        return sinogram.to(ctx.device)

    @staticmethod
    def backward(ctx, grad_sinogram):
        # CUDA A^T
        if ctx.device == 'cuda':
            grad_image = backward_siddon(
                grad_sinogram,
                ctx.image_shape,
                ctx.projections,
                ctx.detectors,
                ctx.detector_spacing,
                ctx.coverage,
                ctx.distance,
                ctx.type
            )
        else:
            grad_image = backward_siddon_cpu(
                grad_sinogram,
                ctx.image_shape,
                ctx.projections,
                ctx.detectors,
                ctx.detector_spacing,
                ctx.coverage,
                ctx.distance,
                ctx.type,
                ctx.workers
            )

        # Gradients corresponding to the inputs of forward()
        return (
            grad_image.to(ctx.device),  # image
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
        workers=16,
        device='cuda',
    ):
        ctx.type = 'f'
        ctx.device = device
        ctx.workers = workers
        # CUDA A^T
        if ctx.device == 'cuda':
            recon_image = backward_siddon(
                sinogram,
                image_shape,
                projections,
                detectors,
                detector_spacing,
                coverage,
                distance,
                ctx.type
            )
        else:
            recon_image = backward_siddon_cpu(
                sinogram,
                image_shape,
                projections,
                detectors,
                detector_spacing,
                coverage,
                distance,
                ctx.type,
                ctx.workers
            )

        # Save what we need to perform A^T later
        ctx.image_shape = image_shape
        ctx.projections = projections
        ctx.detectors = detectors
        ctx.detector_spacing = detector_spacing
        ctx.coverage = coverage
        ctx.distance = distance

        return recon_image.to(ctx.device)

    @staticmethod
    def backward(ctx, grad_recon_image):
        # CUDA A
        if ctx.device == 'cuda':
            grad_sinogram = forward_siddon(
                grad_recon_image.contiguous(),
                ctx.projections,
                ctx.detectors,
                ctx.detector_spacing,
                ctx.coverage,
                ctx.distance,
                ctx.type
            )

        else:
            grad_sinogram = forward_siddon_cpu(
                grad_recon_image.contiguous(),
                ctx.projections,
                ctx.detectors,
                ctx.detector_spacing,
                ctx.coverage,
                ctx.distance,
                ctx.type,
                ctx.workers
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


class FanBeamProjector:
    def __init__(
        self,
        projections=800,
        detector_bins=800,
        detector_spacing=1,
        coverage=2 * torch.pi,
        distance=400,
        type="line",
        device="cpu",
        workers=16,
        pre_weighting=False
    ):
        self.projections = projections
        self.detectors = detector_bins
        self.detector_spacing = detector_spacing
        self.coverage = coverage
        self.distance = distance
        self.type = type
        self.device = device
        self.workers = workers
        self.pre_weighting = pre_weighting

    def forward(self, image):
        self.image_shape = image.shape

        if image.device.type != self.device:
            ValueError(f"Image not on device: {self.device}. ")


        if len(self.image_shape) == 2:
            warnings.warn(
                "Image dimension contains only height and width. "
                "Expected (batch, channels, height, width). "
                "Automatically reshaping to (1, 1, height, width).",
                UserWarning
            )
            self.image_shape = (1, 1, *self.image_shape)
            image = image.unsqueeze(0).unsqueeze(0)

        elif len(self.image_shape) == 3:
            warnings.warn(
                "Image dimension contains three dimensions. "
                "Expected (batch, channels, height, width). "
                "Automatically reshaping to (1, channels, height, width).",
                UserWarning
            )
            self.image_shape = (1, *self.image_shape)
            image = image.unsqueeze(0)

        elif len(self.image_shape) > 4:
            raise ValueError(f"Image dimension is greater than 4, expected 4 dimensions (batch, channels, height, width).")

        if self.type == "line":
            return _FanBeamFunction.apply(
                image,
                self.projections,
                self.detectors,
                self.detector_spacing,
                self.coverage,
                self.distance,
                self.workers,
                self.device
            )

        raise ValueError(f"Unknown projector type: {self.type}")

    # function to perform the backward operation without autograd
    def backward(self, sinogram, image_shape=None):
        self.mode = 'f'
        sinogram_shape = tuple(sinogram.shape)

        if sinogram.device.type != self.device:
            ValueError(f"Image not on device: {self.device}. ")


        if len(sinogram_shape) == 2:
            warnings.warn(
                "Image dimension contains only height and width. "
                "Expected (batch, channels, height, width). "
                "Automatically reshaping to (1, 1, height, width).",
                UserWarning
            )
            sinogram = sinogram.unsqueeze(0).unsqueeze(0)

        elif len(sinogram_shape) == 3:
            warnings.warn(
                "Image dimension contains three dimensions. "
                "Expected (batch, channels, height, width). "
                "Automatically reshaping to (1, channels, height, width).",
                UserWarning
            )
            sinogram = sinogram.unsqueeze(0)

        elif len(sinogram_shape) > 4:
            raise ValueError(f"Sinogram dimension is greater than 4, expected 4 dimensions (batch, channels, height, width).")

        batches = sinogram_shape[0]

        if image_shape is not None:
            self.image_shape = image_shape
            if len(self.image_shape) == 2:
                warnings.warn(
                    "Image dimension contains only height and width. "
                    "Expected (batch, channels, height, width). "
                    "Automatically reshaping to (1, 1, height, width).",
                    UserWarning
                )
                self.image_shape = (batches, 1, *self.image_shape)

            elif len(self.image_shape) == 3:
                warnings.warn(
                    "Image dimension contains three dimensions. "
                    "Expected (batch, channels, height, width). "
                    "Automatically reshaping to (1, channels, height, width).",
                    UserWarning
                )
                self.image_shape = (batches, *self.image_shape)

            elif len(self.image_shape) > 4:
                raise ValueError(f"Image dimension is greater than 4, expected 4 dimensions (batch, channels, height, width).")

        elif not hasattr(self, "image_shape"):
            self.image_shape = (batches, 1, 512, 512)
            raise warnings.warn(
                "Image shape must be provided for backward operation. Will use the default image shape (1x1x512x512).",
                UserWarning
            )
        if sinogram.shape[2] != self.projections or sinogram.shape[3] != self.detectors:
            raise ValueError(
                f"Sinogram shape {sinogram.shape} does not match projector configuration "
                f"({self.projections}, {self.detectors})"
            )

        if self.type == "line":
            if self.device == 'cuda':
                return backward_siddon(
                    sinogram,
                    self.image_shape,
                    self.projections,
                    self.detectors,
                    self.detector_spacing,
                    self.coverage,
                    self.distance,
                    self.mode
                )
            else:
                return backward_siddon_cpu(
                    sinogram,
                    self.image_shape,
                    self.projections,
                    self.detectors,
                    self.detector_spacing,
                    self.coverage,
                    self.distance,
                    self.mode,
                    self.workers
                )

        raise ValueError(f"Unknown projector type: {self.type}")
    
    def _sinc(self, x):
        px = math.pi * x
        zero = x == 0
        safe = torch.where(zero, torch.ones_like(px), px)
        return torch.where(zero, torch.ones_like(x), torch.sin(px) / safe)

    def FBP(self, sinogram, image_shape=None, filter_type="shepp-logan"):

        sinogram_shape = tuple(sinogram.shape)

        if sinogram.device.type != self.device:
            ValueError(f"Image not on device: {self.device}. ")


        if len(sinogram_shape) == 2:
            warnings.warn(
                "Image dimension contains only height and width. "
                "Expected (batch, channels, height, width). "
                "Automatically reshaping to (1, 1, height, width).",
                UserWarning
            )
            sinogram = sinogram.unsqueeze(0).unsqueeze(0)

        elif len(sinogram_shape) == 3:
            warnings.warn(
                "Image dimension contains three dimensions. "
                "Expected (batch, channels, height, width). "
                "Automatically reshaping to (1, channels, height, width).",
                UserWarning
            )
            sinogram = sinogram.unsqueeze(0)

        elif len(sinogram_shape) > 4:
            raise ValueError(f"Sinogram dimension is greater than 4, expected 4 dimensions (batch, channels, height, width).")

        batches = sinogram.shape[0]
        
        if image_shape is not None:
            self.image_shape = image_shape
            if len(self.image_shape) == 2:
                warnings.warn(
                    "Image dimension contains only height and width. "
                    "Expected (batch, channels, height, width). "
                    "Automatically reshaping to (1, 1, height, width).",
                    UserWarning
                )
                self.image_shape = (batches, 1, *self.image_shape)

            elif len(self.image_shape) == 3:
                warnings.warn(
                    "Image dimension contains three dimensions. "
                    "Expected (batch, channels, height, width). "
                    "Automatically reshaping to (1, channels, height, width).",
                    UserWarning
                )
                self.image_shape = (batches, *self.image_shape)

            elif len(self.image_shape) > 4:
                raise ValueError(f"Image dimension is greater than 4, expected 4 dimensions (batch, channels, height, width).")

        elif not hasattr(self, "image_shape"):
            self.image_shape = (batches, 1, 512, 512)
            raise warnings.warn(
                "Image shape must be provided for backward operation. Will use the default image shape (1x1x512x512).",
                UserWarning
            )
        
        if sinogram.shape[2] != self.projections or sinogram.shape[3] != self.detectors:
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

        if self.pre_weighting:
            D = self.distance          # use the source-to-detector distance here
            t = (torch.arange(n, device=sinogram.device) - (n - 1) / 2) * self.detector_spacing
            sinogram = sinogram * (D / torch.sqrt(D**2 + t**2))[None, None, None, :]

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
            F_Rf = torch.fft.fft(sino_padded, dim=3).to(self.device)
            F_Rf_filtered = F_Rf * filt[None, :].to(self.device)
            Rf_filtered = torch.fft.ifft(F_Rf_filtered, dim=3).real[:, :, : , :n].contiguous().to(self.device)

            return _FBPFunction.apply(
                Rf_filtered,
                self.image_shape,
                self.projections,
                self.detectors,
                self.detector_spacing,
                self.coverage,
                self.distance,
                self.workers,
                self.device
            ) * scale

        raise ValueError(f"Unknown projector type: {self.type}")
