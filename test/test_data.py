from difftomo.data import CarvedCheeseDataset
from difftomo.data import SheppLogan
from difftomo.data import WalnutDataset
import pytest

def test_carvedcheese_sinogram_shape():
    dataset = CarvedCheeseDataset()
    assert dataset.sinogram.shape == (45, 989)


def test_carvedcheese_ground_truth_shape():
    dataset = CarvedCheeseDataset()
    assert dataset.gt.shape == (2000, 2000)

def test_carvedcheese_system_matrix_shape():
    dataset = CarvedCheeseDataset()
    assert dataset.matrix.shape == (44414, 65536)


def test_carvedcheese_system_matrix_is_sparse():
    dataset = CarvedCheeseDataset()
    assert dataset.matrix.getformat() == "csc"

def test_carvedcheese_system_matrix_image_dimension():
    dataset = CarvedCheeseDataset()
    A = dataset.matrix
    assert A.shape[1] == 256 * 256


def test_shepp_logan_shape():
    phantom = SheppLogan()
    assert phantom.data.shape == (512, 512)

def test_walnut_dataset_shapes():
    dataset = WalnutDataset()

    true_sino = dataset.sino1200
    true_img = dataset.gt

    res_sino = dataset.sino_res
    res_matrix = dataset.system_matrix

    assert true_sino.shape == (2296, 1200)
    assert true_img.shape == (2296, 2296)

    assert res_sino.shape == (82, 120)
    assert res_matrix.shape == (9840, 6724)


def test_walnut_system_matrix_is_sparse():
    dataset = WalnutDataset()

    assert dataset.system_matrix.is_sparse