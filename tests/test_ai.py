from app.game.ai import GomokuAI


def test_ai_first_move():
    ai = GomokuAI()
    board = [['' for _ in range(15)] for _ in range(15)]
    r, c = ai.get_best_move(board)
    assert r == 7 and c == 7 # Center

def test_forced_block_open_three():
    ai = GomokuAI(depth=2)
    board = [['' for _ in range(15)] for _ in range(15)]
    # Player has open 3 horizontally
    board[5][5] = 'X'
    board[5][6] = 'X'
    board[5][7] = 'X'

    r, c = ai.get_best_move(board)
    # The AI should play at either ends to block: (5,4) or (5,8)
    assert (r == 5 and c == 4) or (r == 5 and c == 8)

def test_forced_block_open_four():
    ai = GomokuAI(depth=2)
    board = [['' for _ in range(15)] for _ in range(15)]
    # Player has open 4 vertically
    board[4][4] = 'X'
    board[5][4] = 'X'
    board[6][4] = 'X'
    board[7][4] = 'X'

    r, c = ai.get_best_move(board)
    # The AI must block at (3,4) or (8,4)
    assert (r == 3 and c == 4) or (r == 8 and c == 4)

def test_winning_move_five():
    ai = GomokuAI(depth=2)
    board = [['' for _ in range(15)] for _ in range(15)]
    # AI has 4 in a row, should complete it
    board[1][1] = 'O'
    board[2][2] = 'O'
    board[3][3] = 'O'
    board[4][4] = 'O'

    # We must ensure there is an empty spot at (0,0) or (5,5)

    r, c = ai.get_best_move(board)
    assert (r == 0 and c == 0) or (r == 5 and c == 5)
