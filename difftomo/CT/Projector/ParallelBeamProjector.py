import torch
from difftomo_cuda import forward_siddon


class ParallelBeamProjector:
    def __init__(
        self,
        projections=800,
        detector_bins=800,
        detector_spacing=1,
        coverage=2 * torch.pi,
        distance=100,
        type="line",
    ):
        self.projections = projections
        self.detectors = detector_bins
        self.detector_spacing = detector_spacing
        self.coverage = coverage
        self.distance = distance
        self.type = type


    def forward(self, image):
        if self.type == "line":
            return forward_siddon(image, 
                                self.projections, 
                                self.detectors, 
                                self.detector_spacing, 
                                self.coverage, 
                                self.distance)

    def backward(self):
        pass
