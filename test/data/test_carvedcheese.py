from difftomo.data import CarvedCheeseDataset


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
