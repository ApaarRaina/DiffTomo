import torch


class SIRT:
    def __init__(self, projector, image_shape=None, iterations=10, device="cpu"):
        self.projector = projector
        self.iterations = iterations
        self.device = device
        self.relaxation_factor = 1
        if image_shape is not None:
            self.image_shape = image_shape
        else:
            self.image_shape = (self.projector.detectors, self.projector.detectors)

    def reconstruct(self, sinogram):

        ones_image = torch.ones((self.projector.detectors, self.projector.detectors), device=self.device)
        row_sum = self.projector.forward(ones_image)
        self.row_sum = torch.where(
                            row_sum > 1e-8,
                            1.0 / row_sum,
                            torch.zeros_like(row_sum)
                        )
        ones_sinogram = torch.ones((self.projector.projections, self.projector.detectors), device=self.device)
        column_sum = self.projector.backward(ones_sinogram, image_shape=self.image_shape)
        self.col_sum = torch.where(
                            column_sum > 1e-8,
                            1.0 / column_sum,
                            torch.zeros_like(column_sum)
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

            #R
            difference = self.row_sum * difference

            # A^T
            correction = self.projector.backward(difference)

            # C
            correction = self.col_sum * correction

            # update
            image += self.relaxation_factor * correction

        return image