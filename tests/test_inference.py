import os
import tempfile

import torch

from app.game.dl_ai import DeepLearningAI
from ml.model import PolicyValueNet


def test_fallback_behavior():
    # Provide a non-existent path
    ai = DeepLearningAI(model_path="non_existent_weights.onnx")

    assert ai.use_fallback is True

    board = [['' for _ in range(15)] for _ in range(15)]
    # Ensure fallback minimax works
    r, c = ai.get_best_move(board)
    assert r == 7 and c == 7 # Fallback minimax plays center

def test_onnx_inference():
    size = 15
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test_model.onnx")

        # Create and export a dummy model
        model = PolicyValueNet(size=size)
        model.eval()
        dummy_input = torch.randn(1, 3, size, size, dtype=torch.float32)

        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'policy': {0: 'batch_size'},
                          'value': {0: 'batch_size'}}
        )

        # Initialize DL AI with the dummy model
        ai = DeepLearningAI(model_path=onnx_path)

        # Verify it loaded correctly and isn't using fallback
        assert ai.use_fallback is False
        assert ai.session is not None

        board = [['' for _ in range(15)] for _ in range(15)]

        # Run inference
        r, c = ai.get_best_move(board)

        assert 0 <= r < size
        assert 0 <= c < size
