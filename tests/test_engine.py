from app.game.engine import GomokuBoard


def test_initial_board():
    board = GomokuBoard(15)
    assert board.size == 15
    assert len(board.board) == 15
    assert all(cell == '' for row in board.board for cell in row)

def test_valid_move():
    board = GomokuBoard()
    assert board.is_valid_move(0, 0) == True
    assert board.is_valid_move(-1, 0) == False
    assert board.is_valid_move(15, 0) == False

    board.make_move(0, 0, 'X')
    assert board.is_valid_move(0, 0) == False

def test_horizontal_win():
    board = GomokuBoard()
    for c in range(4):
        board.make_move(0, c, 'X')
        assert not board.check_win(0, c, 'X')

    board.make_move(0, 4, 'X')
    assert board.check_win(0, 4, 'X')

def test_vertical_win():
    board = GomokuBoard()
    for r in range(4):
        board.make_move(r, 0, 'O')
        assert not board.check_win(r, 0, 'O')

    board.make_move(4, 0, 'O')
    assert board.check_win(4, 0, 'O')

def test_diagonal_win():
    board = GomokuBoard()
    for i in range(4):
        board.make_move(i, i, 'X')
        assert not board.check_win(i, i, 'X')

    board.make_move(4, 4, 'X')
    assert board.check_win(4, 4, 'X')
