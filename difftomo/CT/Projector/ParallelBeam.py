import torch
from ._extensions import forward_siddon, backward_siddon


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
            return forward_siddon(
                image,
                self.projections,
                self.detectors,
                self.detector_spacing,
                self.coverage,
                self.distance,
            )

    def backward(self, sinogram):
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
