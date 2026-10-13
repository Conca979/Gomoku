import torch

from ml.encoder import GomokuEncoder
from ml.model import PolicyValueNet


def test_model_shapes():
    # 1. Test encoder shape
    encoder = GomokuEncoder(size=15)
    board = [['' for _ in range(15)] for _ in range(15)]
    board[7][7] = 'X'
    tensor = encoder.encode_board(board, 'X')

    assert tensor.shape == (3, 15, 15)
    assert tensor[0][7][7] == 1.0 # Current player (X)
    assert tensor[1][7][7] == 0.0 # Opponent
    assert tensor[2][0][0] == 1.0 # Color to play (X)

    # 2. Test model input/output shapes
    net = PolicyValueNet(size=15)

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 15, 15)

    policy_logits, value = net(dummy_input)

    assert policy_logits.shape == (batch_size, 225)
    assert value.shape == (batch_size, 1)

    # Check value bounds
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)
