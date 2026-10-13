import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from ml.encoder import GomokuEncoder
from ml.model import PolicyValueNet


def simulate_self_play(net: PolicyValueNet, encoder: GomokuEncoder, size: int = 15, num_games: int = 1) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Simulate a few dummy self-play games for demonstration.
    In reality, this would use MCTS to generate target policy pi and value z.
    Returns: list of (board_tensor, target_pi, target_z)
    """
    data = []
    for _ in range(num_games):
        # Dummy generation
        board = [['' for _ in range(size)] for _ in range(size)]

        # Simulate some random moves and states
        for _ in range(5):
            current_player = 'X' if np.random.rand() > 0.5 else 'O'
            tensor = encoder.encode_board(board, current_player)

            # Dummy target policy (uniform)
            target_pi = np.ones(size * size, dtype=np.float32) / (size * size)

            # Dummy target value
            target_z = float(np.random.choice([-1.0, 0.0, 1.0]))

            data.append((tensor, target_pi, target_z))
    return data

def train(epochs: int = 1, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    size = 15
    net = PolicyValueNet(size=size).to(device)
    encoder = GomokuEncoder(size=size)

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4) # L2 regularization (c||theta||^2)

    # Generate dummy training data
    raw_data = simulate_self_play(net, encoder, size, num_games=10)

    # Apply symmetries to expand dataset
    train_data = []
    for state, pi, z in raw_data:
        syms = encoder.get_symmetries(state, pi)
        for s_state, s_pi in syms:
            train_data.append((s_state, s_pi, z))

    print(f"Generated {len(train_data)} training samples (including symmetries).")

    for epoch in range(epochs):
        net.train()

        # Shuffle
        np.random.shuffle(train_data) # type: ignore

        total_loss = 0.0
        batches = len(train_data) // batch_size
        if batches == 0:
            batches = 1
            batch_size = len(train_data)

        for i in range(batches):
            batch = train_data[i * batch_size:(i + 1) * batch_size]

            states = torch.FloatTensor(np.array([item[0] for item in batch])).to(device)
            target_pis = torch.FloatTensor(np.array([item[1] for item in batch])).to(device)
            target_zs = torch.FloatTensor(np.array([item[2] for item in batch])).unsqueeze(1).to(device)

            optimizer.zero_grad()

            policy_logits, values = net(states)

            # Loss = (z - v)^2 - pi^T log(p) + c||theta||^2
            # Weight decay is handled by optimizer
            value_loss = F.mse_loss(values, target_zs)

            # Use log_softmax for numerical stability
            log_p = F.log_softmax(policy_logits, dim=1)
            # Cross entropy: - sum(pi * log(p))
            policy_loss = -torch.mean(torch.sum(target_pis * log_p, dim=1))

            loss = value_loss + policy_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/batches:.4f}")

    # Save checkpoint
    torch.save(net.state_dict(), "gomoku_model.pt")
    print("Model saved to gomoku_model.pt")

if __name__ == "__main__":
    train()
