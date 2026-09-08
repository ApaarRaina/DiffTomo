from ..utils import download_file, extract_zip
from pathlib import Path
from scipy.io import loadmat
import torch
import numpy as np

class WalnutDataset:
    def __init__(self, res=82):
        self.path = f"Walnut_dataset/Data{res}.mat"
        self.gt_path = f"Walnut_dataset/GroundTruthReconstruction.mat"
        self.og_path = f"Walnut_dataset/FullSizeSinograms.mat"
        self.url, self.destination = 'https://zenodo.org/api/records/1254206/files-archive', 'Walnut_dataset'
        self.downloaded = download_file(self.url, "files_archive")
        self.unzipped = extract_zip(Path("files_archive"), Path(self.destination))
        self.system_matrix = self._extract_system_matrix(self.path)
        self.sino_res = self._extract_sinogram(self.path)
        self.sino120 = self._get_og_data(self.og_path)
        self.sino1200 = self._get_og_data(self.og_path)
        self.gt = self._get_gt_data(self.gt_path)

    def _extract_system_matrix(self, path):
        data = loadmat(path)
        data = data['A']
        data = data.tocoo()

        indices = torch.tensor(np.array([data.row, data.col]), dtype=torch.long)
        values = torch.tensor(np.array(data.data), dtype=torch.float32)

        return torch.sparse_coo_tensor(indices, values, size=data.shape)

    def _extract_sinogram(self, path):
        data = loadmat(path)
        data = data['m']

        return torch.tensor(data)

    def _get_gt_data(self, path):
        data = loadmat(path)
        data = data['FBP1200']

        return torch.from_numpy(data.astype(np.float32))

    def _get_og_data(self, path):
        data = loadmat(path)
        data = data["sinogram1200"]
        return torch.from_numpy(data.astype(np.float32))
