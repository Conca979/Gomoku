# Gomoku AI Strategy

## Algorithm: Minimax with Alpha-Beta Pruning
The Gomoku AI relies on the **Minimax** algorithm to traverse the game tree. The algorithm simulates all possible moves up to a certain `depth` (currently restricted to 2 plies to maintain sub-second response times). It operates on the assumption that both the AI (maximizing player) and the human (minimizing player) play optimally.

To vastly improve performance, the algorithm is enhanced with **Alpha-Beta pruning**. This optimization stops evaluating a branch in the tree once it guarantees that the current player has a better move elsewhere.

## Search Space Bounding
In a standard 15x15 Gomoku board, there are 225 possible moves initially, leading to an extremely high branching factor.
To mitigate this, our engine employs **Search Bounding**:
- We only consider squares that are within a distance of 1 or 2 from already occupied squares.
- This immediately cuts down the number of candidates per ply and focuses the AI on relevant tactical fights.

## Heuristic Evaluation Function
When the search hits the depth limit (or a leaf node), the state is evaluated using a heuristic scoring function. The evaluation works by assigning points based on threat formations:
- **Set of 5**: 100,000 pts (Game won)
- **Open 4**: 10,000 pts (Guarantee a win on the next turn)
- **Half-open 4**: 1,000 pts
- **Open 3**: 1,000 pts (Highly dangerous, must be blocked)
- **Half-open 3**: 100 pts
- **Open 2**: 100 pts
- **Half-open 2**: 10 pts

To ensure the AI prioritizes defensive blocks when necessary, the heuristic evaluates the score for the AI and subtracts the score of the human player, multiplied by a slight defensive bias (e.g., 1.1x multiplier for human threats).
