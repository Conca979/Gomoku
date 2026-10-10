import math


class GomokuAI:
    def __init__(self, ai_player: str = 'O', human_player: str = 'X', depth: int = 2):
        self.ai = ai_player
        self.human = human_player
        self.depth = depth

    def get_best_move(self, board_matrix: list[list[str]]) -> tuple[int, int]:
        size = len(board_matrix)

        # If board is empty, play in the center
        is_empty = all(board_matrix[r][c] == '' for r in range(size) for c in range(size))
        if is_empty:
            return size // 2, size // 2

        best_score = -math.inf
        best_move = (-1, -1)

        candidates = self._get_candidate_moves(board_matrix)

        # Alpha-beta pruning root
        alpha = -math.inf
        beta = math.inf

        for r, c in candidates:
            board_matrix[r][c] = self.ai
            if self._check_win(board_matrix, r, c, self.ai):
                board_matrix[r][c] = ''
                return r, c # Winning move, take it immediately
            score = self._minimax(board_matrix, self.depth - 1, alpha, beta, False, r, c)
            board_matrix[r][c] = ''

            if score > best_score:
                best_score = score
                best_move = (r, c)
            alpha = max(alpha, best_score)

        return best_move

    def _minimax(self, board: list[list[str]], depth: int, alpha: float, beta: float, is_maximizing: bool, last_r: int, last_c: int) -> float:
        player = self.ai if not is_maximizing else self.human
        # Check if the previous move ended the game
        if self._check_win(board, last_r, last_c, player):
            return 1000000 if player == self.ai else -1000000

        if depth == 0:
            return self._evaluate_board(board)

        candidates = self._get_candidate_moves(board)
        if not candidates:
            return 0 # Draw

        if is_maximizing:
            max_eval = -math.inf
            for r, c in candidates:
                board[r][c] = self.ai
                eval_score = self._minimax(board, depth - 1, alpha, beta, False, r, c)
                board[r][c] = ''
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for r, c in candidates:
                board[r][c] = self.human
                eval_score = self._minimax(board, depth - 1, alpha, beta, True, r, c)
                board[r][c] = ''
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _get_candidate_moves(self, board: list[list[str]]) -> list[tuple[int, int]]:
        size = len(board)
        candidates: set[tuple[int, int]] = set()

        # Check adjacent squares (distance 1 and 2) around existing pieces
        for r in range(size):
            for c in range(size):
                if board[r][c] != '':
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == '':
                                candidates.add((nr, nc))

        # Sort candidates using a shallow evaluation to improve alpha-beta pruning efficiency
        # For a full AI, we could order by heuristic, but here we just return a list
        return list(candidates)

    def _evaluate_board(self, board: list[list[str]]) -> float:
        score = 0.0
        score += self._evaluate_player(board, self.ai)
        score -= self._evaluate_player(board, self.human) * 1.1 # slightly prioritize blocking
        return score

    def _evaluate_player(self, board: list[list[str]], player: str) -> float:
        size = len(board)
        score = 0.0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(size):
            for c in range(size):
                if board[r][c] == player:
                    for dr, dc in directions:
                        score += self._evaluate_direction(board, r, c, dr, dc, player)

        # _evaluate_direction evaluates each line segment multiple times
        # (once from each piece in the line). A more optimal way would be
        # to scan lines rather than from points, but this suffices for the
        # depth constraint.
        return score

    def _evaluate_direction(self, board: list[list[str]], r: int, c: int, dr: int, dc: int, player: str) -> float:
        size = len(board)

        # Only evaluate if this is the "start" of a sequence to avoid double counting
        prev_r, prev_c = r - dr, c - dc
        if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r][prev_c] == player:
            return 0.0

        count = 1
        curr_r, curr_c = r + dr, c + dc
        while 0 <= curr_r < size and 0 <= curr_c < size and board[curr_r][curr_c] == player:
            count += 1
            curr_r += dr
            curr_c += dc

        # Check bounds at ends
        open_ends = 0

        # Check end 1 (before start)
        if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r][prev_c] == '':
            open_ends += 1

        # Check end 2 (after end)
        if 0 <= curr_r < size and 0 <= curr_c < size and board[curr_r][curr_c] == '':
            open_ends += 1

        if count >= 5:
            return 100000
        elif count == 4:
            if open_ends == 2:
                return 10000
            elif open_ends == 1:
                return 1000
        elif count == 3:
            if open_ends == 2:
                return 1000  # open 3 is powerful
            elif open_ends == 1:
                return 100
        elif count == 2:
            if open_ends == 2:
                return 100
            elif open_ends == 1:
                return 10
        elif count == 1 and open_ends == 2:
            return 1

        return 0.0

    def _check_win(self, board: list[list[str]], row: int, col: int, player: str) -> bool:
        if row < 0 or col < 0:
            return False

        size = len(board)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False
