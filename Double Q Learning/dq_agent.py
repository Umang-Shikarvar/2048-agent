import pickle
import os

class DQAgent:
    def __init__(self, path="dql_qtable_alpha_0.15.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.Q1, self.Q2 = pickle.load(f)
        else:
            self.Q1, self.Q2 = {}, {}
            print("[DQAgent] Warning: No Double-Q table found!")

    def act(self, game):
        s = tuple(game.board)
        legal = game.legal_moves()
        # Sum Q1 + Q2 (Double Q-Learning policy)
        values = [self.Q1.get((s, a), 0) + self.Q2.get((s, a), 0) for a in legal]
        return legal[values.index(max(values))]