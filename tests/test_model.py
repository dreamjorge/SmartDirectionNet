import torch

from smartdirectionnet.model import DirectionClassifier


def test_direction_classifier_forward_returns_one_logit_per_row():
    model = DirectionClassifier(input_dim=4, hidden_sizes=(8,))

    logits = model(torch.randn(5, 4))

    assert logits.shape == (5,)


def test_direction_classifier_default_hidden_sizes_build_expected_layer_count():
    model = DirectionClassifier(input_dim=3)

    linear_layers = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]

    # input->32, 32->16, 16->1
    assert [layer.out_features for layer in linear_layers] == [32, 16, 1]
