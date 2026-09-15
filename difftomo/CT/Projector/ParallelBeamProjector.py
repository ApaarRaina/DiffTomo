import torch

class ParralelBeamProject():

    def __init__(self, projections=800, detector_bins=800, detector_spacing=1, coverage=torch.pi, type='line'):
        self.projections = projections
        self.detectors = detector_bins
        self.detector_spacing = detector_spacing
        self.coverage = coverage
        self.type = type

    def foward():
        pass