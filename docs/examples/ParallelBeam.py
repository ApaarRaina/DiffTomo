from difftomo.CT.Projector import ParallelBeamProjector
from difftomo.data import SheppLogan
import matplotlib.pyplot as plt

img = SheppLogan().data.to("cuda")

projector = ParallelBeamProjector(device="cuda")
sinogram = projector.forward(img)

backprojection = projector.backward(sinogram)

fbp_img = projector.FBP(sinogram)


plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(img.cpu().numpy(), cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(sinogram.cpu().detach().numpy(), cmap='gray')
plt.title("Sinogram")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(backprojection.cpu().detach().numpy(), cmap='gray')
plt.title("Backprojection")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(fbp_img.cpu().detach().numpy(), cmap='gray')
plt.title("FBP Reconstruction")
plt.axis("off")

plt.show()