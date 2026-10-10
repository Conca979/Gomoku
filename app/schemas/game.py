from pydantic import BaseModel, Field


class GameState(BaseModel):
    board: list[list[str]] = Field(..., description="15x15 board. Empty is '', 'X' is Player, 'O' is AI")
    player_turn: str = Field(..., description="'X' or 'O'")
    last_move: tuple[int, int] | None = Field(None, description="The last move played")

class MoveResponse(BaseModel):
    row: int = Field(..., description="Row index of the move (0-14)")
    col: int = Field(..., description="Column index of the move (0-14)")
    is_win: bool = Field(..., description="Whether this move results in a win")
    message: str = Field("", description="Optional message")
