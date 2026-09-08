from difftomo.data import WalnutDataset


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
