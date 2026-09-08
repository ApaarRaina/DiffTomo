import numpy as np
import torch
from pathlib import Path

class SheppLogan:
    def __init__(self):
        self.path = Path(__file__).resolve().parents[2] / "difftomo" / "data" / "image" / "shepp_logan.npy"
        self.data = torch.tensor(np.load(self.path))