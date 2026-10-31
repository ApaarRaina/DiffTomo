from difftomo.CT.Projector import FanBeamProjector
from difftomo.data import SheppLogan
import matplotlib.pyplot as plt

img = SheppLogan().data.to("cuda")

projector = FanBeamProjector(projections=800, detector_bins=800, device="cuda")
sinogram = projector.forward(img)

backprojection = projector.backward(sinogram)

fbp_img = projector.FBP(sinogram, filter_type="ram-lak")
fbp_img_shepp = projector.FBP(sinogram, filter_type="shepp-logan")
fbp_img_cosine = projector.FBP(sinogram, filter_type="cosine")
fbp_img_hamming = projector.FBP(sinogram, filter_type="hamming")
fbp_img_hann = projector.FBP(sinogram, filter_type="hann")

fig, axes = plt.subplots(3, 3, figsize=(15, 9))

axes[0, 0].imshow(img.cpu(), cmap="gray")
axes[0, 0].set_title("Original Image")

axes[0, 1].imshow(sinogram.cpu(), cmap="gray")
axes[0, 1].set_title("Sinogram")

axes[0, 2].imshow(backprojection.cpu(), cmap="gray")
axes[0, 2].set_title("Backprojection")

axes[1, 0].imshow(fbp_img.cpu(), cmap="gray")
axes[1, 0].set_title("FBP (Ram-Lak)")

axes[1, 1].imshow(fbp_img_shepp.cpu(), cmap="gray")
axes[1, 1].set_title("FBP (Shepp-Logan)")

axes[1, 2].imshow(fbp_img_cosine.cpu(), cmap="gray")
axes[1, 2].set_title("FBP (Cosine)")

axes[2, 0].imshow(fbp_img_hamming.cpu(), cmap="gray")
axes[2, 0].set_title("FBP (Hamming)")

axes[2, 1].imshow(fbp_img_hann.cpu(), cmap="gray")
axes[2, 1].set_title("FBP (Hann)")

plt.show()
