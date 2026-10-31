from difftomo.CT.Iterative import DRPDenoising
import torch
import matplotlib.pyplot as plt
from difftomo.data import SheppLogan
from difftomo.CT.Projector import ParallelBeamProjector, FanBeamProjector


def to_np(t):
    return t.detach().cpu().squeeze().numpy()


device = "cuda"
img = SheppLogan().data.to(device)

# Paper: epochs=250, inner_steps=200, lr=5e-4, step_size=0.25
epochs, inner_steps, step_size = 200, 100, 0.25


###########################################################################
#                         PARALLEL BEAM GEOMETRY                          #
###########################################################################

projector_sparse = ParallelBeamProjector(
    projections=100, detector_bins=900, device=device
)
sino_sparse = projector_sparse.forward(img)
fbp_sparse = projector_sparse.FBP(sino_sparse)

drp = DRPDenoising(
    projector_sparse, epochs=epochs, inner_steps=inner_steps, step_size=step_size
)
recon_drp = drp.reconstruct(sino_sparse)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Parallel Beam Geometry", fontsize=16)

axes[0].imshow(to_np(fbp_sparse), cmap="gray")
axes[0].set_title("FBP, sparse-view")

axes[1].imshow(to_np(recon_drp), cmap="gray")
axes[1].set_title("DRP, sparse-view")

axes[2].imshow(to_np(img), cmap="gray")
axes[2].set_title("Ground truth")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


###########################################################################
#                         FAN BEAM GEOMETRY                               #
###########################################################################

projector_sparse = FanBeamProjector(projections=100, detector_bins=900, device=device)
sino_sparse = projector_sparse.forward(img)
fbp_sparse = projector_sparse.FBP(sino_sparse)

drp = DRPDenoising(
    projector_sparse, epochs=epochs, inner_steps=inner_steps, step_size=step_size
)
recon_drp = drp.reconstruct(sino_sparse)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Fan Beam Geometry", fontsize=16)

axes[0].imshow(to_np(fbp_sparse), cmap="gray")
axes[0].set_title("FBP, sparse-view")

axes[1].imshow(to_np(recon_drp), cmap="gray")
axes[1].set_title("DRP, sparse-view")

axes[2].imshow(to_np(img), cmap="gray")
axes[2].set_title("Ground truth")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
