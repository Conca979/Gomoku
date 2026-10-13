import os

import torch

from ml.model import PolicyValueNet


def export_onnx(pt_path: str = "gomoku_model.pt", onnx_path: str = "weights/gomoku_model.onnx"):
    size = 15
    model = PolicyValueNet(size=size)

    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found. Train the model first.")
        # Create a dummy model for testing purposes if it doesn't exist
        print("Creating a dummy ONNX model for testing...")
    else:
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))

    model.eval()

    # Ensure weights directory exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Dummy input
    dummy_input = torch.randn(1, 3, size, size, dtype=torch.float32)

    # Export
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
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
