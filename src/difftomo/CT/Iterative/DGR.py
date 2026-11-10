import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import warnings


class DGR:

    def __init__(self, projector, device="cuda"):
        self.projector = projector
        self.device = device

    def create_B(self, box_size, del_mu):
        kh, kw = box_size[2], box_size[3]

        dx = torch.arange(-(kw // 2), kw // 2 + 1, device=self.device)
        dy = torch.arange(-(kh // 2), kh // 2 + 1, device=self.device)

        dy, dx = torch.meshgrid(dy, dx, indexing="ij")

        B = torch.stack((dx, dy), dim=-1)

        return B.unsqueeze(0) - del_mu[:, None, None, :]

    def compute_D(self, B, covariances):
        variances = covariances.square()

        return (
            B[..., 0].square() + B[..., 1].square()
        ) / variances[:, None, None]

    def tv_loss(self, image):
        dx = image[..., :, 1:] - image[..., :, :-1]
        dy = image[..., 1:, :] - image[..., :-1, :]

        return torch.sqrt(
            dx[..., :-1, :].square()
            + dy[..., :, :-1].square()
            + 1e-6
        ).mean()

    def scatter_add(self, recon_img, means, Gamma, box_size):

        H, W = recon_img.shape[-2:]
        kh, kw = box_size

        # Integer center of each Gaussian
        centers = torch.floor(means).long()

        # Local box coordinates
        dx = torch.arange(-(kw // 2), kw // 2 + 1, device=self.device)
        dy = torch.arange(-(kh // 2), kh // 2 + 1, device=self.device)

        dy, dx = torch.meshgrid(dy, dx, indexing='ij')

        # Global coordinates
        x = centers[:, 0, None, None] + dx
        y = centers[:, 1, None, None] + dy

        # Flatten
        x = x.reshape(-1)
        y = y.reshape(-1)
        Gamma = Gamma.reshape(-1)

        # Boundary mask
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)

        # Convert to flattened image indices
        indices = y[valid] * W + x[valid]

        # Accumulate
        recon_img.view(-1).scatter_add_(
            0,
            indices,
            Gamma[valid]
        )

        return recon_img


    def reconstruct(self, sinogram, image_size, box_size=(9, 9), lamdas=(1.0, 0.6), no_of_estimators=1.5e5, iterations=2e4):

        if len(image_size) == 2:
            warnings.warn("Image size is 2D Expected size to be in 4D of shape (B, C, H, W) will go by (1, 1, H, W)", UserWarning)
            image_size = (1, 1, image_size[0], image_size[1])

        elif len(image_size) == 3:
            warnings.warn("Image size is 3D Expected size to be in 4D of shape (B, C, H, W) will go by (1, C, H, W)", UserWarning)
            image_size = (1, image_size[0], image_size[1], image_size[2])

        box_size = (image_size[0], image_size[1], box_size[0], box_size[1])

        #initialise gaussian means
        no_of_estimators = int(no_of_estimators)
        means = torch.rand((no_of_estimators, 2), requires_grad=True, device=self.device)
        with torch.no_grad():
            means[:, 0] *= image_size[-1]
            means[:, 1] *= image_size[-2]

        #initialise gaussian covariances
        covariances = torch.full((no_of_estimators,), 1.0, requires_grad=True, device=self.device)

        #initialise intensities
        raw_intensities = torch.rand(no_of_estimators, device=self.device, requires_grad=True)

        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam([means, covariances, raw_intensities], lr=5e-4)

        pbar = tqdm(range(int(iterations)))

        for i in pbar:
            canvas_img = torch.zeros(image_size, device=self.device)
            del_mu = means - torch.floor(means)
            B = self.create_B(box_size, del_mu)
            D = self.compute_D(B, covariances)
            intensities = F.softplus(raw_intensities)
            Gamma = torch.exp(-0.5 * D) * intensities[:, None, None]

            recon_img = self.scatter_add(
                            canvas_img,
                            means,
                            Gamma,
                            box_size[-2:]
                        )

            recon_proj = self.projector.forward(recon_img)
            data_loss = mse_loss(recon_proj, sinogram)
            TV_loss = self.tv_loss(recon_img)

            loss = (lamdas[0] * data_loss + lamdas[1] * TV_loss).mean()

            pbar.set_postfix({
                "Data": f"{data_loss.item():.6f}",
                "TV": f"{TV_loss.item():.6f}",
                "Total": f"{loss.item():.6f}"
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            canvas_img = torch.zeros(image_size, device=self.device)

            del_mu = means - torch.floor(means)
            B = self.create_B(box_size, del_mu)
            D = self.compute_D(B, covariances)
            intensities = F.softplus(raw_intensities)   
            Gamma = torch.exp(-0.5 * D) * intensities[:, None, None]

            recon_img = self.scatter_add(
                            canvas_img,
                            means,
                            Gamma,
                            box_size[-2:]
                        )

        return recon_img

            