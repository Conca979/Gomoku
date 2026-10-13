import os

import numpy as np

from app.game.ai import GomokuAI

try:
    import onnxruntime as ort  # type: ignore
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class DeepLearningAI:
    def __init__(self, ai_player: str = 'O', human_player: str = 'X', model_path: str = "weights/gomoku_model.onnx"):
        self.ai = ai_player
        self.human = human_player
        self.size = 15
        self.fallback_ai = GomokuAI(ai_player=ai_player, human_player=human_player, depth=2)

        self.session: ort.InferenceSession | None = None # type: ignore
        self.use_fallback = False

        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available. Falling back to Minimax.")
            self.use_fallback = True
        elif not os.path.exists(model_path):
            print(f"Model {model_path} not found. Falling back to Minimax.")
            self.use_fallback = True
        else:
            try:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            except Exception as e: # noqa: BLE001
                print(f"Error loading ONNX model: {e}. Falling back to Minimax.")
                self.use_fallback = True

    def encode_board(self, board: list[list[str]]) -> np.ndarray:
        tensor = np.zeros((1, 3, self.size, self.size), dtype=np.float32)

        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == self.ai:
                    tensor[0][0][r][c] = 1.0 # AI stone
                elif board[r][c] == self.human:
                    tensor[0][1][r][c] = 1.0 # Human stone

        # Third channel is color to play (AI is playing, let's say AI=0)
        # Using 0 to represent O (AI) to be consistent with encoder logic where X is 1.
        if self.ai == 'X':
            tensor[0][2][:][:] = 1.0

        return tensor

    def get_best_move(self, board_matrix: list[list[str]]) -> tuple[int, int]:
        if self.use_fallback or self.session is None:
            return self.fallback_ai.get_best_move(board_matrix)

        # Neural Network inference
        tensor = self.encode_board(board_matrix)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})

        policy_logits = outputs[0][0] # (225,)

        # Apply softmax to get probabilities
        exp_preds = np.exp(policy_logits - np.max(policy_logits))
        policy_probs = exp_preds / np.sum(exp_preds)

        # Mask invalid moves
        valid_moves_mask = np.zeros(self.size * self.size, dtype=np.float32)
        has_valid_moves = False
        for r in range(self.size):
            for c in range(self.size):
                if board_matrix[r][c] == '':
                    valid_moves_mask[r * self.size + c] = 1.0
                    has_valid_moves = True

        if not has_valid_moves:
            return -1, -1

        policy_probs = policy_probs * valid_moves_mask

        sum_probs = np.sum(policy_probs)
        if sum_probs > 0:
            policy_probs /= sum_probs
        else:
            # Fallback if network outputs extremely low logits for all valid moves
            return self.fallback_ai.get_best_move(board_matrix)

        # Select move with highest probability
        best_move_idx = int(np.argmax(policy_probs))
        r = best_move_idx // self.size
        c = best_move_idx % self.size

        return r, c
