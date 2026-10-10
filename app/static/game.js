const BOARD_SIZE = 15;
let board = [];
let gameActive = true;
let playerTurn = 'X'; // Human plays first

const boardElement = document.getElementById('board');
const statusElement = document.getElementById('status');
const resetBtn = document.getElementById('reset-btn');

function initGame() {
    board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
    gameActive = true;
    playerTurn = 'X';
    statusElement.textContent = 'Your Turn (X)';
    statusElement.style.color = '';
    renderBoard();
}

function renderBoard() {
    boardElement.innerHTML = '';
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            if (board[r][c] !== '') {
                cell.classList.add('occupied');
                cell.classList.add(board[r][c].toLowerCase());
                cell.textContent = board[r][c];
            }
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.addEventListener('click', handleCellClick);
            boardElement.appendChild(cell);
        }
    }
}

async function handleCellClick(event) {
    if (!gameActive || playerTurn !== 'X') return;

    const r = parseInt(event.target.dataset.row);
    const c = parseInt(event.target.dataset.col);

    if (board[r][c] !== '') return;

    // Player move
    board[r][c] = 'X';
    renderBoard();

    statusElement.textContent = 'AI is thinking...';
    playerTurn = 'O'; // Lock UI while waiting

    try {
        const response = await fetch('/api/move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                board: board,
                player_turn: 'O',
                last_move: [r, c]
            })
        });

        const data = await response.json();

        if (data.is_win) {
            gameActive = false;
            statusElement.textContent = data.message;
            if (data.row !== -1 && data.col !== -1) {
                board[data.row][data.col] = 'O';
                renderBoard();
            }
            return;
        }

        if (data.row === -1 && data.col === -1) {
            gameActive = false;
            statusElement.textContent = "Draw!";
            return;
        }

        // AI move
        board[data.row][data.col] = 'O';
        renderBoard();

        statusElement.textContent = 'Your Turn (X)';
        playerTurn = 'X';

    } catch (error) {
        console.error("Error making API call:", error);
        statusElement.textContent = 'Error connecting to server';
        gameActive = false;
    }
}

resetBtn.addEventListener('click', initGame);

// Initialize on load
initGame();
