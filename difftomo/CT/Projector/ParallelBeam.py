import torch
from ._extensions import forward_siddon, backward_siddon



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
            None,        # projections
            None,        # detectors
            None,        # detector_spacing
            None,        # coverage
            None,        # distance
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
    ):
        self.projections = projections
        self.detectors = detector_bins
        self.detector_spacing = detector_spacing
        self.coverage = coverage
        self.distance = distance
        self.type = type

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
            raise Warning("Image shape must be provided for backward operation. Will use the default image shape (512x512).")
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
