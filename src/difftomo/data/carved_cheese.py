from ..utils import download_file, load_mat
from pathlib import Path
import torch
import numpy as np
from scipy.sparse import csc_matrix
import h5py


class CarvedCheeseDataset:
    def __init__(self, res=256, projections=45, view="Full"):
        self.resolution = res
        self.projections = projections
        self.view = view

        self.destination = Path("CarvedCheese")

        self.url_base = "https://zenodo.org/records/1254210/files"

        # File required for this particular configuration
        self.data_filename = f"Data{self.view}_{self.resolution}x{self.projections}.mat"

        self.url_list = [
            f"{self.url_base}/{self.data_filename}?download=1",
            f"{self.url_base}/FullSizeSinograms.mat?download=1",
            f"{self.url_base}/GroundTruthReconstruction.mat?download=1",
        ]

        self.downloaded = self._download_folder(
            self.url_list,
            self.destination,
        )

        self.sinogram = self._extract_sinogram(
            self.resolution,
            self.projections,
            self.view,
        )

        self.matrix = self._extract_system_matrix(
            self.resolution,
            self.projections,
            self.view,
        )

        self.gt = self._extract_gt()

    def _download_folder(self, url_list, destination):
        destination.mkdir(parents=True, exist_ok=True)

        for url in url_list:
            filename = Path(url.split("?")[0]).name
            output_path = destination / filename

            if output_path.exists():
                print(f"{filename} already exists. Skipping download.")
                continue

            print(f"Downloading {filename}...")
            download_file(url, output_path)

        return 1

    def _extract_sinogram(self, res, projections, view):
        path = self.destination / f"Data{view}_{res}x{projections}.mat"
        return torch.tensor(load_mat(path)["m"])

    def extract_measured_sinograms(self, projections=15, view="Full"):
        path = self.destination / "FullSizeSinograms.mat"
        return torch.tensor(load_mat(path)[f"sinogram{projections}{view}View"])

    def _extract_norm(self, view, res, projections):
        path = self.destination / f"Data{view}_{res}x{projections}.mat"
        return torch.tensor(load_mat(path)["normA"])

    def _extract_system_matrix(self, res, projections, view):
        filename = f"Data{view}_{res}x{projections}.mat"
        path = Path(self.destination) / filename

        with h5py.File(path, "r") as f:
            A = f["A"]

            data = np.array(A["data"]).squeeze()
            ir = np.array(A["ir"]).squeeze().astype(np.int64)
            jc = np.array(A["jc"]).squeeze().astype(np.int64)

            return csc_matrix((data, ir, jc))

    def _extract_gt(self):
        path = self.destination / "GroundTruthReconstruction.mat"
        return torch.tensor(load_mat(path)["FBP360"])
