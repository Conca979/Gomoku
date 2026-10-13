from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.game.dl_ai import DeepLearningAI
from app.game.engine import GomokuBoard
from app.schemas.game import GameState, MoveResponse

app = FastAPI(title="Gomoku AI")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.post("/api/move", response_model=MoveResponse)
async def make_ai_move(state: GameState):
    board = state.board
    size = len(board)

    if size != 15 or any(len(row) != 15 for row in board):
        raise HTTPException(status_code=400, detail="Board must be 15x15")

    engine = GomokuBoard(size=size)
    engine.board = [row[:] for row in board]  # Copy

    ai = DeepLearningAI(ai_player='O', human_player='X')

    if state.last_move:
        lr, lc = state.last_move
        if engine.check_win(lr, lc, 'X'):
            return MoveResponse(row=-1, col=-1, is_win=True, message="Player wins")

    r, c = ai.get_best_move(engine.board)

    if r == -1 or c == -1:
        return MoveResponse(row=-1, col=-1, is_win=False, message="Draw")

    engine.make_move(r, c, 'O')
    is_win = engine.check_win(r, c, 'O')

    message = "AI wins" if is_win else ""
    return MoveResponse(row=r, col=c, is_win=is_win, message=message)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
