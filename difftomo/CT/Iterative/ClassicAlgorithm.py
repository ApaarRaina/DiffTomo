import torch
import warnings
from difftomo.CT.Projector._extensions import ARTReconstruction


class ART:
    def __init__(self, projector, iterations=10, relaxation_factor=1):
        self.projector = projector
        self.iterations = iterations
        self.relaxation_factor = relaxation_factor

    def reconstruct(self, sinogram, image_shape=None):
        if image_shape is not None:
            self.image_shape = image_shape
        else:
            warnings.warn(
                "Image shape not specified. Defaulting to (1, 1, 256, 256). Please specify the image shape for accurate reconstruction.",
                UserWarning,
            )
            self.image_shape = (1, 1, 256, 256)

        reconstructed_image = ARTReconstruction(
            sinogram,
            self.image_shape,
            self.projector.projections,
            self.projector.detectors,
            self.iterations,
            self.relaxation_factor,
            self.projector.detector_spacing,
            self.projector.coverage,
            self.projector.distance,
            self.projector.mode,
        )
        return reconstructed_image


class SIRT:
    def __init__(self, projector, iterations=10, device="cpu"):
        self.projector = projector
        self.iterations = iterations
        self.device = device
        self.relaxation_factor = 1

    def reconstruct(self, sinogram, image_shape=None):
        if image_shape is not None:
            self.image_shape = image_shape
        else:
            self.image_shape = (1, 1, 512, 512)
            warnings.warn(
                "No image shape specified gonna take default shape of 512x512",
                UserWarning,
            )

        if len(sinogram.shape) == 4:
            batches, channels = sinogram.shape[0], sinogram.shape[1]
        elif len(sinogram.shape) == 3:
            batches, channels = 1, sinogram.shape[0]
        elif len(sinogram.shape) == 2:
            batches, channels = 1, 1
        else:
            ValueError("Invalid sinogram shape. Expected 2D, 3D, or 4D tensor.")

        self.image_shape = (batches, channels, self.image_shape[0], self.image_shape[1])

        ones_image = torch.ones(
            (batches, channels, self.projector.detectors, self.projector.detectors),
            device=self.device,
        )
        row_sum = self.projector.forward(ones_image)
        self.row_sum = torch.where(
            row_sum > 1e-8, 1.0 / row_sum, torch.zeros_like(row_sum)
        )
        ones_sinogram = torch.ones(
            (batches, channels, self.projector.projections, self.projector.detectors),
            device=self.device,
        )
        column_sum = self.projector.backward(
            ones_sinogram, image_shape=self.image_shape
        )
        self.col_sum = torch.where(
            column_sum > 1e-8, 1.0 / column_sum, torch.zeros_like(column_sum)
        )
        # Initialize the image with zeros
        image_shape = self.image_shape
        image = torch.zeros(image_shape, dtype=torch.float32)
        image = image.to(self.device)

        # Perform SIRT iterations
        for i in range(self.iterations):
            # Forward projections
            projected = self.projector.forward(image)
            projected = projected.to(self.device)

            # Compute the difference between the measured and projected sinogram
            difference = sinogram - projected

            # Backward projection of the difference
            difference = difference.to(self.device)

            # Update the image

            # R
            difference = self.row_sum * difference

            # A^T
            correction = self.projector.backward(difference)

            # C
            correction = self.col_sum * correction

            # update
            image += self.relaxation_factor * correction

        return image


class SART:
    def __init__(self):
        pass
