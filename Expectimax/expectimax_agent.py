# expectimax_agent.py
import math, random

# ========================= HEURISTIC EVALUATION =========================

# Strong 2048 heuristic weights
WEIGHTS = {
    "empty":      270,
    "max_corner": 10000,
    "monotonic":  47,
    "smooth":     0.1,
}

# Snake-like ordering (encourages monotonic high-value placement)
SNAKE = [
    [65536, 32768, 16384, 8192],
    [512,   1024,  2048, 4096],
    [256,   128,    64,   32],
    [2,       4,      8,   16]
]

def evaluate(board):
    """Heuristic: monotonicity + smoothness + empty tiles + max-corner bonus."""
    tiles = [(1 << v) if v > 0 else 0 for v in board]
    g = [tiles[i*4:(i+1)*4] for i in range(4)]

    # 1) empty tiles
    empty = sum(1 for t in tiles if t == 0)

    # 2) max tile in corner
    max_tile = max(tiles)
    corner_bonus = (g[3][3] == max_tile)

    # 3) monotonic (weighted snake pattern)
    mono = sum(SNAKE[r][c] * g[r][c] for r in range(4) for c in range(4))

    # 4) smoothness (minimize local gradient)
    smooth = 0
    for r in range(4):
        for c in range(3):
            smooth -= abs(g[r][c] - g[r][c+1])
    for c in range(4):
        for r in range(3):
            smooth -= abs(g[r][c] - g[r+1][c])

    return (
        WEIGHTS["empty"] * empty +
        WEIGHTS["max_corner"] * corner_bonus +
        WEIGHTS["monotonic"] * mono +
        WEIGHTS["smooth"] * smooth
    )

# ============================= EXPECTIMAX ===============================

class ExpectimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def act(self, game):
        """Return best action 0,1,2,3 based on Expectimax search."""
        best, _ = self.maximize(game, self.depth)
        return best

    # --------- MAX NODE (Player Chooses Move) ---------
    def maximize(self, game, depth):
        if depth == 0:
            return None, evaluate(game.board)

        legal = game.legal_moves()
        if not legal:
            return None, -float('inf')

        best, best_val = None, -float('inf')
        for a in legal:
            clone = copy_game(game)
            clone.move(a)
            val = self.expect(clone, depth - 1)  # FIXED DEPTH REDUCTION

            if val > best_val:
                best_val = val
                best = a

        return best, best_val

    # --------- CHANCE NODE (Random Tile Appears) ---------
    def expect(self, game, depth):
        if depth == 0:
            return evaluate(game.board)

        empty_indices = [i for i, v in enumerate(game.board) if v == 0]
        if not empty_indices:
            return evaluate(game.board)

        total = 0
        for i in empty_indices:
            for tile, prob in [(1, 0.9), (2, 0.1)]:  # new tile: 2 or 4
                clone = copy_game(game)
                clone.board[i] = tile
                _, val = self.maximize(clone, depth - 1)
                total += prob * val

        return total / len(empty_indices)

# ====================== HELPER FOR GAME COPY ============================

def copy_game(game):
    from env import Game2048  # local import to avoid circular import
    g = Game2048()
    g.board = game.board[:]
    g.score = game.score
    return g