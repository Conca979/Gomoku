class GomokuBoard:
    def __init__(self, size: int = 15):
        self.size = size
        self.board = [['' for _ in range(size)] for _ in range(size)]

    def is_valid_move(self, row: int, col: int) -> bool:
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.board[row][col] == ''
        return False

    def make_move(self, row: int, col: int, player: str) -> bool:
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            return True
        return False

    def undo_move(self, row: int, col: int) -> None:
        if 0 <= row < self.size and 0 <= col < self.size:
            self.board[row][col] = ''

    def check_win(self, row: int, col: int, player: str) -> bool:
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1)   # diagonal down-left
        ]

        for dr, dc in directions:
            count = 1
            # Check forward
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Check backward
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False
