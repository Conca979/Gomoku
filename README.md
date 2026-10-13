# Gomoku AI Minimalist

A playable, full-stack Gomoku (Five-in-a-Row) game featuring a Player vs AI mode. The frontend features a modern, minimalist white design using 'X' and 'O'. The backend provides an AI engine that uses a Minimax algorithm with Alpha-Beta pruning and heuristic evaluation to provide challenging gameplay.

For a deep dive into the AI approach, please see [AI_STRATEGY.md](AI_STRATEGY.md).

## Prerequisites
- Python >= 3.10
- `uv` (Fast Python package and project manager)

## Setup and Installation

This project strictly uses `uv` for environment management and script executions.

1. Install dependencies and set up the virtual environment:
   ```bash
   uv sync
   ```

## Running the Game

Start the FastAPI application via Uvicorn:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, open your browser and navigate to `http://localhost:8000` or `http://127.0.0.1:8000` to play the game!

## Running Tests and Validation

The project enforces code quality, type safety, and unit testing. Run the following verification commands using `uv`:

1. Run Ruff for linting and formatting checks:
   ```bash
   uv run ruff check .
   ```

2. Run Mypy for static type checking:
   ```bash
   uv run mypy app/
   ```

3. Run Pytest for unit testing (engine and AI heuristics):
   ```bash
   PYTHONPATH=. uv run pytest tests/ -v
   ```
