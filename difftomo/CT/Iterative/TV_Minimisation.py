import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TVDenoising:
    def __init__(self, projector, image_shape=None, iterations=10, lamda=0.1, device="cpu"):
        self.projector = projector
        self.iterations = iterations
        self.device = device
        self.lr = 0.05
        self.lam = lamda
        if image_shape is not None:
            self.image_shape = image_shape
        else:
            self.image_shape = (self.projector.detectors, self.projector.detectors)

    def sobel_filter(self, image):
        sobel_x = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32).to(self.device)
        sobel_y = torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32).to(self.device)

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        grad_x = nn.functional.conv2d(image.unsqueeze(0), sobel_x, padding=1)
        grad_y = nn.functional.conv2d(image.unsqueeze(0), sobel_y, padding=1)

        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def reconstruct(self, noisy_img):
        noisy_img = noisy_img.to(self.device)
        image = noisy_img.clone().to(self.device)
        image.requires_grad_(True)
        projected_img = self.projector.forward(noisy_img)

        for _ in range(self.iterations):
            D_f = torch.mean((projected_img - self.projector.forward(image))**2)
            R_x = torch.sum(torch.abs(self.sobel_filter(image)))
            l = D_f + self.lam * R_x
            l.backward()
            with torch.no_grad():
                image -= self.lr * image.grad
            image.grad.zero_()
        
        return image