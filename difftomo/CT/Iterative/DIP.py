import torch
import torch.nn as nn
from difftomo.CT.Models import PriorModel
from tqdm import tqdm


class DRPDenoising:
    def __init__(self, projector, epochs=250, inner_steps=50, step_size=0.25, lr=5e-4):
        self.projector = projector
        self.epochs = epochs
        self.inner_steps = inner_steps
        self.beta = step_size
        self.lr = lr
        self.loss = nn.MSELoss()

    def reconstruct(self, sparse_sinogram, prior_model=None):
        device = self.projector.device
        p = sparse_sinogram.to(device)

        if prior_model is None:
            prior = PriorModel().to(device)
        else:
            prior = prior_model

        optimizer = torch.optim.Adam(prior.parameters(), lr=self.lr)

        z = self.projector.FBP(p).detach()

        for epoch in tqdm(range(self.epochs)):
            with torch.no_grad():
                x = prior(z)
                delta = self.projector.backward(p - self.projector.forward(x))
                z = x + self.beta * delta

            for _ in range(self.inner_steps):
                loss = self.loss(self.projector.forward(prior(z)), p)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            return prior(z)
