import random
from copy import deepcopy
import json

class SarsaNTupleAgent:
    def __init__(self, n_tuples=None, max_tile_value=15, alpha=0.01, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_tile = max_tile_value

        if n_tuples is None:
            # default 4-tuples: rows, columns and 2x2 squares (common setup)
            self.n_tuples = [
                [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15],
                [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15],
                [0,1,4,5], [1,2,5,6], [2,3,6,7],
                [4,5,8,9], [5,6,9,10], [6,7,10,11],
                [8,9,12,13], [9,10,13,14], [10,11,14,15]
            ]
        else:
            self.n_tuples = n_tuples

        self.base = self.max_tile + 1
        self.tables = []
        for tpl in self.n_tuples:
            size = (self.base ** len(tpl))
            self.tables.append([0.0] * size)

    # encode tuple -> index
    def _encode(self, board, tpl):
        key = 0
        m = 1
        for idx in tpl:
            v = board[idx]
            if v > self.max_tile:
                v = self.max_tile
            key += v * m
            m *= self.base
        return key

    # estimate V(s) as mean of tuple LUTs
    def v(self, board):
        s = 0.0
        for tbl, tpl in zip(self.tables, self.n_tuples):
            k = self._encode(board, tpl)
            s += tbl[k]
        return s / len(self.tables)

    # update V(s) by adding alpha * delta to corresponding LUT entries
    def _update_v(self, board, delta):
        for tbl, tpl in zip(self.tables, self.n_tuples):
            k = self._encode(board, tpl)
            tbl[k] += self.alpha * delta

    # epsilon-greedy over afterstates: pick action with max (r + V(afterstate))
    def act(self, game):
        legal = game.legal_moves()
        if not legal:
            return None
        if random.random() < self.epsilon:
            return random.choice(legal)

        best_val = -1e12
        best_a = None
        for a in legal:
            temp = deepcopy(game)
            r = temp.move(a)
            if r < 0:
                continue
            after = temp.board
            val = r + self.v(after)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    # SARSA update in afterstate representation:
    # target = reward + gamma * V(next_after)  (if next exists)
    def learn_step(self, s_after, reward, s_next_after, next_action_exists):
        v_s = self.v(s_after)
        if next_action_exists and (s_next_after is not None):
            target = reward + self.gamma * self.v(s_next_after)
        else:
            target = reward
        delta = target - v_s
        self._update_v(s_after, delta)

    def set_epsilon(self, e):
        self.epsilon = e

    def set_alpha(self, a):
        self.alpha = a

    def save(self, path):
        payload = {"n_tuples": self.n_tuples, "base": self.base, "tables": self.tables}
        with open(path, "w") as f:
            json.dump(payload, f)

    def load(self, path):
        with open(path, "r") as f:
            payload = json.load(f)
        self.n_tuples = payload["n_tuples"]
        self.base = payload["base"]
        self.tables = payload["tables"]
