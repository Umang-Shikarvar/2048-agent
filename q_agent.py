import pickle

class QAgent:
    def __init__(self, path="qtable_alpha_0.15.pkl"):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)

    def act(self, game):
        s = tuple(game.board)
        legal = game.legal_moves()
        values = [self.Q.get((s, a), 0) for a in legal]
        return legal[values.index(max(values))]