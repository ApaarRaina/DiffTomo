from difftomo.CT.Iterative import SIRT, ART, TVDenoising, TGVDenoising, SART
import torch
import matplotlib.pyplot as plt
from difftomo.data import SheppLogan
from difftomo.CT.Projector import ParallelBeamProjector, FanBeamProjector
import cv2 as cv


###########################################################################
#                         PARALLEL BEAM GEOMETRY                          #
###########################################################################


device = "cuda"
iterations = 150
img = SheppLogan().data.to(device)

# ---- full-view setup ----
projector_full = ParallelBeamProjector(device=device)
sino_full = projector_full.forward(img)
fbp_full = projector_full.FBP(sino_full)

# ---- sparse-view setup ----
projector_sparse = ParallelBeamProjector(
    projections=100, detector_bins=800, device=device
)
sino_sparse = projector_sparse.forward(img)
fbp_sparse = projector_sparse.FBP(sino_sparse)

# ---- reconstructions ----
sirt_full = SIRT(projector_full, iterations=iterations, device=device)
recon_sirt_full = sirt_full.reconstruct(sino_full, image_shape=img.shape)

sirt_sparse = SIRT(projector_sparse, iterations=iterations, device=device)
recon_sirt_sparse = sirt_sparse.reconstruct(
    sino_sparse,
    image_shape=img.shape,
)

sart_full = SART(projector_full, iterations=iterations, device=device)
recon_sart_full = sart_full.reconstruct(sino_full, image_shape=img.shape)

sart_sparse = SART(projector_sparse, iterations=iterations, device=device)
recon_sart_sparse = sart_sparse.reconstruct(
    sino_sparse,
    image_shape=img.shape,
)

# TV: use the SPARSE projector, since the data term compares A·x to the sparse sinogram
tv = TVDenoising(projector_sparse, iterations=20, lamda=0.05, device=device)
recon_tv_sparse = tv.reconstruct(fbp_sparse)  # (sinogram, initial image) signature

# TGV: use the SPARSE projector, since the data term compares A·x to the sparse sinogram
tgv = TGVDenoising(projector_sparse, iterations=20, lamda=[1, 2], device=device)
recon_tgv_sparse = tgv.reconstruct(fbp_sparse)  # (sinogram, initial image) signature


# ---- plotting ----
def to_np(t):
    return t.squeeze(0).squeeze(0).detach().cpu().numpy()


vmin, vmax = img.min().item(), img.max().item()  # shared range for image panels

fig, axes = plt.subplots(2, 6, figsize=(15, 9))
fig.suptitle("Parallel Beam Geometry", fontsize=16)

# Row 1: full-view (800 projections)
axes[0, 0].imshow(to_np(sino_full), cmap="gray")
axes[0, 0].set_title(f"Full sinogram ({projector_full.projections} views)")

axes[0, 1].imshow(to_np(fbp_full), cmap="gray")
axes[0, 1].set_title(f"FBP, full-view")

axes[0, 2].imshow(to_np(recon_sirt_full), cmap="gray")
axes[0, 2].set_title(f"SIRT, full-view ({iterations} it.)")

axes[0, 3].imshow(to_np(recon_sart_full), cmap="gray")
axes[0, 3].set_title(f"SART, full-view ({iterations} it.)")

axes[0, 4].imshow(to_np(img), cmap="gray")
axes[0, 4].set_title("Ground truth")

# Row 2: sparse-view (100 projections)
axes[1, 0].imshow(to_np(sino_sparse), cmap="gray")
axes[1, 0].set_title(f"Sparse sinogram ({projector_sparse.projections} views)")

axes[1, 1].imshow(to_np(fbp_sparse), cmap="gray")
axes[1, 1].set_title(f"FBP, sparse-view")

axes[1, 2].imshow(to_np(recon_sirt_sparse), cmap="gray")
axes[1, 2].set_title(f"SIRT, sparse-view ({iterations} it.)")

axes[1, 3].imshow(to_np(recon_sart_sparse), cmap="gray")
axes[1, 3].set_title(f"SART, sparse-view ({iterations} it.)")

axes[1, 4].imshow(to_np(recon_tv_sparse), cmap="gray")
axes[1, 4].set_title(f"TV, sparse-view ({20} it.)")

axes[1, 5].imshow(to_np(recon_tgv_sparse), cmap="gray")
axes[1, 5].set_title(f"TGV, sparse-view ({20} it.)")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()


###########################################################################
#                         FAN BEAM GEOMETRY                               #
###########################################################################

device = "cuda"
iterations = 150
img = SheppLogan().data.to(device)

# ---- full-view setup ----
projector_full = FanBeamProjector(projections=1000, detector_bins=900, device=device)
sino_full = projector_full.forward(img)
fbp_full = projector_full.FBP(sino_full)

# ---- sparse-view setup ----
projector_sparse = FanBeamProjector(projections=100, detector_bins=900, device=device)
sino_sparse = projector_sparse.forward(img)
fbp_sparse = projector_sparse.FBP(sino_sparse)

# ---- reconstructions ----
sirt_full = SIRT(projector_full, iterations=iterations, device=device)
recon_sirt_full = sirt_full.reconstruct(sino_full, image_shape=img.shape)

sirt_sparse = SIRT(projector_sparse, iterations=iterations, device=device)
recon_sirt_sparse = sirt_sparse.reconstruct(sino_sparse, image_shape=img.shape)

sart_full = SART(projector_full, iterations=iterations, device=device)
recon_sart_full = sart_full.reconstruct(sino_full, image_shape=img.shape)

sart_sparse = SART(projector_sparse, iterations=iterations, device=device)
recon_sart_sparse = sart_sparse.reconstruct(
    sino_sparse,
    image_shape=img.shape,
)

# TV: use the SPARSE projector, since the data term compares A·x to the sparse sinogram
tv = TVDenoising(projector_sparse, iterations=20, lamda=0.05, device=device)
recon_tv_sparse = tv.reconstruct(fbp_sparse)  # (sinogram, initial image) signature

# TGV: use the SPARSE projector, since the data term compares A·x to the sparse sinogram
tgv = TGVDenoising(projector_sparse, iterations=20, lamda=[1, 2], device=device)
recon_tgv_sparse = tgv.reconstruct(fbp_sparse)  # (sinogram, initial image) signature


vmin, vmax = img.min().item(), img.max().item()  # shared range for image panels

fig, axes = plt.subplots(2, 6, figsize=(15, 9))
fig.suptitle("Fan Beam Geometry", fontsize=16)

# Row 1: full-view (800 projections)
axes[0, 0].imshow(to_np(sino_full), cmap="gray")
axes[0, 0].set_title(f"Full sinogram ({projector_full.projections} views)")

axes[0, 1].imshow(to_np(fbp_full), cmap="gray")
axes[0, 1].set_title(f"FBP, full-view")

axes[0, 2].imshow(to_np(recon_sirt_full), cmap="gray")
axes[0, 2].set_title(f"SIRT, full-view ({iterations} it.)")

axes[0, 3].imshow(to_np(recon_sart_full), cmap="gray")
axes[0, 3].set_title(f"SART, full-view ({iterations} it.)")

axes[0, 4].imshow(to_np(img), cmap="gray")
axes[0, 4].set_title("Ground truth")

# Row 2: sparse-view (100 projections)
axes[1, 0].imshow(to_np(sino_sparse), cmap="gray")
axes[1, 0].set_title(f"Sparse sinogram ({projector_sparse.projections} views)")

axes[1, 1].imshow(to_np(fbp_sparse), cmap="gray")
axes[1, 1].set_title(f"FBP, sparse-view")

axes[1, 2].imshow(to_np(recon_sirt_sparse), cmap="gray")
axes[1, 2].set_title(f"SIRT, sparse-view ({iterations} it.)")

axes[1, 3].imshow(to_np(recon_sart_sparse), cmap="gray")
axes[1, 3].set_title(f"SART, sparse-view ({iterations} it.)")

axes[1, 4].imshow(to_np(recon_tv_sparse), cmap="gray")
axes[1, 4].set_title(f"TV, sparse-view ({20} it.)")

axes[1, 5].imshow(to_np(recon_tgv_sparse), cmap="gray")
axes[1, 5].set_title(f"TGV, sparse-view ({20} it.)")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()


# specifically for ART
###########################################################################
#                         PARALLEL BEAM GEOMETRY                          #
###########################################################################


phantom = SheppLogan()
projector = ParallelBeamProjector(projections=300, detector_bins=500, device="cuda")
img = phantom.data.to("cuda")
img = torch.tensor(cv.resize(img.cpu().numpy(), (200, 200)), device="cuda")
sinogram = projector.forward(img)
fbp_img = projector.FBP(sinogram)

sinogram = sinogram.to("cpu")
art_recon = ART(projector, iterations=10).reconstruct(sinogram, image_shape=(200, 200))

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(img.cpu().numpy(), cmap="gray")
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(fbp_img.squeeze(0).squeeze(0).cpu().numpy(), cmap="gray")
plt.title("FBP Reconstructed Image")

plt.subplot(1, 3, 3)
plt.imshow(art_recon.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap="gray")
plt.title("ART Reconstruction")

plt.show()


###########################################################################
#                         FAN BEAM GEOMETRY                               #
###########################################################################

phantom = SheppLogan()
projector = FanBeamProjector(projections=300, detector_bins=500, device="cuda")
img = phantom.data.to("cuda")
img = torch.tensor(cv.resize(img.cpu().numpy(), (200, 200)), device="cuda")
sinogram = projector.forward(img)
fbp_img = projector.FBP(sinogram)

sinogram = sinogram.to("cpu")
art = ART(projector, iterations=10)
art_recon = art.reconstruct(sinogram, image_shape=(200, 200))

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(img.cpu().numpy(), cmap="gray")
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(fbp_img.squeeze(0).squeeze(0).cpu().numpy(), cmap="gray")
plt.title("FBP Reconstructed Image")

plt.subplot(1, 3, 3)
plt.imshow(art_recon.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap="gray")
plt.title("ART Reconstruction")

plt.show()
