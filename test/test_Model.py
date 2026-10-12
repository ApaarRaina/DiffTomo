from difftomo.CT.Models import REDCNN, Reconstructor
import torch
import pytest


@pytest.mark.parametrize("load_pretrained", [True])
def test_redcnn_weights(load_pretrained):
    model_A = REDCNN(
        load_pretrained=load_pretrained,
        device="cpu",
    )

    A = [p.detach().clone() for p in model_A.parameters()]

    model_B = REDCNN(
        load_pretrained=load_pretrained,
        device="cpu",
    )

    B = [p.detach().clone() for p in model_B.parameters()]

    model_Random = REDCNN(
        load_pretrained=False,
        device="cpu",
    )

    random = [p.detach().clone() for p in model_Random.parameters()]

    A, B, random = torch.tensor(A[0]), torch.tensor(B[0]), torch.tensor(random[0])

    if load_pretrained:
        assert torch.allclose(A, B, atol=1e-6), "Weights should be equal when loading pretrained model."
        assert not torch.allclose(A, random, atol=1e-6), "Weights should not be equal to random model when loading pretrained model."

@pytest.mark.parametrize("load_pretrained", [True])
def test_reconstructor_weights(load_pretrained):
    model_A = Reconstructor(
        img_h=512,
        img_w=512,
        load_pretrained=load_pretrained,
        device="cpu",
    )

    A = model_A.Image_recon_module.conv_first.weight.detach().clone()

    model_B = Reconstructor(
        img_h=512,
        img_w=512,
        load_pretrained=load_pretrained,
        device="cpu",
    )
    
    B = model_B.Image_recon_module.conv_first.weight.detach()

    model_random = Reconstructor(
        img_h=512,
        img_w=512,
        load_pretrained=False,
        device="cpu",
    )

    random = model_random.Image_recon_module.conv_first.weight.detach()

    if load_pretrained:
        assert torch.allclose(A, B, atol=1e-6), "Weights should be equal when loading pretrained model."
        assert not torch.allclose(A, random, atol=1e-6), "Weights should not be equal to random model when loading pretrained model."