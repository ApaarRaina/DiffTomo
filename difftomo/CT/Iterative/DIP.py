import torch
import torch.nn as nn
from difftomo.CT.Models import PriorModel



class DRPDenoising:
    def __init__(self, projector, iterations=30, step_size=0.1):
        self.projector = projector
        self.iterations = iterations
        self.Prior = PriorModel().to(self.projector.device)
        self.loss = nn.MSELoss()
        self.beat = step_size

    def reconstruct(self, sparse_sinogram):

        sparse_sinogram = sparse_sinogram.to(self.projector.device)
        z = self.projector.FBP(sparse_sinogram)
        residual = sparse_sinogram - self.projector.forward(self.Prior(z.unsqueeze(0).unsqueeze(0)))
        grad = self.projector.backward(residual)
        l = self.loss(self.projector.forward(self.Prior(z.unsqueeze(0).unsqueeze(0))), sparse_sinogram)
        optimizer = torch.optim.Adam(self.Prior.parameters(), lr=1e-4)

        for i in range(self.iterations):
            z = self.Prior(z.unsqueeze(0).unsqueeze(0)) + self.beat * grad
            l = self.loss(self.projector.forward(self.Prior(z.unsqueeze(0).unsqueeze(0))), sparse_sinogram)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            residual = sparse_sinogram - self.projector.forward(self.Prior(z.unsqueeze(0).unsqueeze(0)))
            grad = self.projector.backward(residual)

        return self.Prior(z)



        

