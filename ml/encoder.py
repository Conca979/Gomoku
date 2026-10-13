
import numpy as np


class GomokuEncoder:
    def __init__(self, size: int = 15):
        self.size = size

    def encode_board(self, board: list[list[str]], current_player: str) -> np.ndarray:
        """
        Encode the board into a 3x15x15 tensor.
        Channel 0: 1 if current_player stone, 0 otherwise
        Channel 1: 1 if opponent stone, 0 otherwise
        Channel 2: 1 everywhere if current_player == 'X' (or just 1 everywhere to indicate color to play)
                   For simplicity, let's use 1 if current_player is 'X', 0 if 'O'.
        """
        tensor = np.zeros((3, self.size, self.size), dtype=np.float32)

        opponent = 'O' if current_player == 'X' else 'X'

        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == current_player:
                    tensor[0][r][c] = 1.0
                elif board[r][c] == opponent:
                    tensor[1][r][c] = 1.0

        # Third channel represents color to play. 1 for X, 0 for O.
        if current_player == 'X':
            tensor[2, :, :] = 1.0

        return tensor

    def get_symmetries(self, board_tensor: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate 8 dihedral symmetries of the board and corresponding policy probabilities.
        pi is a flat array of size (size * size)
        """
        pi_board = np.reshape(pi, (self.size, self.size))
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board_tensor, i, axes=(1, 2))
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)

                symmetries.append((new_b, new_pi.flatten()))

        return symmetries
