import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math


class TVDenoising:
    def __init__(
        self, projector, image_shape=None, iterations=10, lamda=0.1, device="cpu"
    ):
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
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).to(self.device)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).to(self.device)

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
            D_f = torch.mean((projected_img - self.projector.forward(image)) ** 2)
            R_x = torch.sum(torch.abs(self.sobel_filter(image)))
            l = D_f + self.lam * R_x
            l.backward()
            with torch.no_grad():
                image -= self.lr * image.grad
            image.grad.zero_()

        return image


class TGVDenoising:
    """
    Second-order TGV using the Chambolle-Pock primal-dual algorithm.

    Objective (with A replaced by the normalized operator A/L, L = ||A||):
        min_{u,w} 0.5 * ||A(u)/L - b/L||^2
                   + alpha1 * ||grad(u) - w||_{1,2}
                   + alpha2 * ||E(w)||_{1,F}

    w: 2-channel vector field
    E(w): symmetric gradient, with the off-diagonal component scaled by
          1/sqrt(2) so its Euclidean norm equals the Frobenius norm.

    Assumes projector.forward(x) computes A(x) and
    projector.backward(y) computes the adjoint A^T(y).

    Note: because A is normalized, alpha1/alpha2 are image-scale weights.
    This is equivalent to using alpha * L^2 in the un-normalized problem.
    """

    def __init__(
        self,
        projector,
        image_shape=None,
        iterations=300,
        lamda=None,
        device="cpu",
        tau=None,
        sigma=None,
        theta=1.0,
        normalize=True,
        verbose=False,
    ):
        self.projector = projector
        self.image_shape = image_shape
        self.iterations = iterations
        self.device = torch.device(device)
        self.theta = theta
        self.normalize = normalize
        self.verbose = verbose

        # lamda = [alpha1, alpha2]
        if lamda is None:
            self.alpha1, self.alpha2 = 0.1, 0.1
        elif isinstance(lamda, (int, float)):
            self.alpha1 = self.alpha2 = float(lamda)
        else:
            self.alpha1, self.alpha2 = map(float, lamda)

        self.tau = tau
        self.sigma = sigma

        self.L = 1.0  # operator norm of A, set in reconstruct()

    def _A(self, x):
        return self.projector.forward(x) / self.L

    def _AT(self, y):
        return self.projector.backward(y) / self.L

    @staticmethod
    def Dx(x):
        out = torch.zeros_like(x)
        out[..., :-1] = x[..., 1:] - x[..., :-1]
        return out

    @staticmethod
    def Dy(x):
        out = torch.zeros_like(x)
        out[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
        return out

    @staticmethod
    def DxT(y):
        out = torch.zeros_like(y)
        out[..., 0] = -y[..., 0]
        out[..., 1:-1] = y[..., :-2] - y[..., 1:-1]
        out[..., -1] = y[..., -2]
        return out

    @staticmethod
    def DyT(y):
        out = torch.zeros_like(y)
        out[..., 0, :] = -y[..., 0, :]
        out[..., 1:-1, :] = y[..., :-2, :] - y[..., 1:-1, :]
        out[..., -1, :] = y[..., -2, :]
        return out

    def grad(self, u):
        # Shape: [2, H, W]
        return torch.stack((self.Dx(u), self.Dy(u)), dim=0)

    def grad_adjoint(self, p):
        return self.DxT(p[0]) + self.DyT(p[1])

    def epsilon(self, w):
        wx, wy = w[0], w[1]

        e_xx = self.Dx(wx)
        e_yy = self.Dy(wy)
        e_xy = (self.Dy(wx) + self.Dx(wy)) / math.sqrt(2.0)

        # Shape: [3, H, W]
        return torch.stack((e_xx, e_yy, e_xy), dim=0)

    def epsilon_adjoint(self, q):
        q_xx, q_yy, q_xy = q[0], q[1], q[2]
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        wx = self.DxT(q_xx) + inv_sqrt2 * self.DyT(q_xy)
        wy = self.DyT(q_yy) + inv_sqrt2 * self.DxT(q_xy)

        return torch.stack((wx, wy), dim=0)

    @staticmethod
    def project_ball(x, radius):
        norm = torch.sqrt(torch.sum(x * x, dim=0, keepdim=True))
        scale = torch.clamp(norm / radius, min=1.0)
        return x / scale

    @torch.no_grad()
    def estimate_A_norm(self, image_shape, n_iter=30):
        u = torch.randn(image_shape, device=self.device, dtype=torch.float32)
        u = u / torch.linalg.norm(u)

        eig = 1.0
        for _ in range(n_iter):
            v = self.projector.backward(self.projector.forward(u))
            eig = torch.linalg.norm(v).clamp_min(1e-12)  # ~ ||A||^2
            u = v / eig

        return torch.sqrt(eig).item()

    @torch.no_grad()
    def estimate_operator_norm(self, image_shape, n_iter=30):
        u = torch.randn(image_shape, device=self.device, dtype=torch.float32)
        w = torch.randn((2, *image_shape), device=self.device, dtype=u.dtype)

        norm = torch.sqrt(torch.sum(u * u) + torch.sum(w * w))
        u, w = u / norm, w / norm

        for _ in range(n_iter):
            r = self._A(u)
            p = self.grad(u) - w
            q = self.epsilon(w)

            # K^T K (u, w) = K^*(r, p, q)
            u_next = self._AT(r) + self.grad_adjoint(p)
            w_next = -p + self.epsilon_adjoint(q)

            norm = torch.sqrt(
                torch.sum(u_next * u_next) + torch.sum(w_next * w_next)
            ).clamp_min(1e-12)

            u, w = u_next / norm, w_next / norm

        r = self._A(u)
        p = self.grad(u) - w
        q = self.epsilon(w)

        norm_K = torch.sqrt(torch.sum(r * r) + torch.sum(p * p) + torch.sum(q * q))
        return norm_K.item()

    @torch.no_grad()
    def reconstruct(self, noisy_img):
        noisy_img = noisy_img.to(self.device, dtype=torch.float32)

        # Normalize the projector so alpha1/alpha2 are image-scale weights
        self.L = self.estimate_A_norm(noisy_img.shape) if self.normalize else 1.0

        # noisy_img is an image; its (normalized) projections are the
        # data-fidelity target.
        b = self._A(noisy_img)

        u = noisy_img.clone()
        w = torch.zeros((2, *u.shape), device=self.device, dtype=u.dtype)

        # Dual variables: data term, first-order TGV, second-order TGV
        r = torch.zeros_like(b)
        p = torch.zeros_like(w)
        q = torch.zeros((3, *u.shape), device=self.device, dtype=u.dtype)

        # Step sizes satisfying tau*sigma*||K||^2 < 1.
        # Power iteration underestimates ||K||, so add a 5% margin.
        norm_K = 1.05 * self.estimate_operator_norm(u.shape)
        if self.tau is None or self.sigma is None:
            step = 0.95 / max(norm_K, 1e-8)
            tau = sigma = step
        else:
            tau, sigma = self.tau, self.sigma
            if tau * sigma * norm_K**2 >= 1:
                raise ValueError("Step sizes violate tau * sigma * ||K||^2 < 1.")

        if self.verbose:
            print(
                f"[TGV] L={self.L:.4f}  ||K||={norm_K:.4f}  "
                f"tau={tau:.4f}  sigma={sigma:.4f}"
            )

        u_bar = u.clone()
        w_bar = w.clone()

        for it in range(self.iterations):
            # 1. Data dual update:
            # prox_{sigma F*}(r + sigma*(A(u_bar) - b)), F(z)=0.5||z-b||^2
            r = (r + sigma * (self._A(u_bar) - b)) / (1.0 + sigma)

            # 2. First-order TGV dual update
            p = p + sigma * (self.grad(u_bar) - w_bar)
            p = self.project_ball(p, self.alpha1)

            # 3. Second-order TGV dual update
            q = q + sigma * self.epsilon(w_bar)
            q = self.project_ball(q, self.alpha2)

            # 4. Primal image update
            u_old = u
            w_old = w

            u = u - tau * (self._AT(r) + self.grad_adjoint(p))

            # 5. Primal auxiliary-field update
            w = w + tau * p - tau * self.epsilon_adjoint(q)

            # 6. Extrapolation
            u_bar = u + self.theta * (u - u_old)
            w_bar = w + self.theta * (w - w_old)

            if self.verbose and (it % 10 == 0 or it == self.iterations - 1):
                p_sat = (p.norm(dim=0).max() / self.alpha1).item()
                q_sat = (q.norm(dim=0).max() / self.alpha2).item()
                print(
                    f"[TGV] it={it:4d}  ||u-u0||={torch.linalg.norm(u - noisy_img).item():.4f}  "
                    f"p/alpha1={p_sat:.3f}  q/alpha2={q_sat:.3f}"
                )

        return u
