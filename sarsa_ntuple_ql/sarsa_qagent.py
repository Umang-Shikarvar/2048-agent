import random
import pickle
from copy import deepcopy

class SarsaQNTupleAgent:
    """
    Q(s,a) approximated by n-tuple lookup tables.
    For each action a we maintain a set of tuple LUTs. Q(s,a) = mean_over_tuples LUT_a[t][key(afterstate)].
    The board uses log2 values (0 empty, 1->2, 2->4...).
    """
    def __init__(self, n_tuples=None, max_tile_value=15, alpha=0.01, gamma=0.99, epsilon=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_tile = max_tile_value

        # default tuples: rows, columns and 2x2 squares (same as earlier)
        if n_tuples is None:
            self.n_tuples = [
                [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15],      # rows
                [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15],      # cols
                [0,1,4,5], [1,2,5,6], [2,3,6,7], [4,5,8,9],
                [5,6,9,10], [6,7,10,11], [8,9,12,13], [9,10,13,14], [10,11,14,15]
            ]
        else:
            self.n_tuples = n_tuples

        self.base = self.max_tile + 1
        self.num_actions = 4

        # tables[action][tuple_index] -> flattened list of weights
        self.tables = []
        for _ in range(self.num_actions):
            action_tables = []
            for tpl in self.n_tuples:
                size = (self.base ** len(tpl))
                action_tables.append([0.0] * size)
            self.tables.append(action_tables)


    def _encode(self, board, tpl):
        key = 0
        m = 1
        for idx in tpl:
            v = board[idx]
            if v > self.max_tile: v = self.max_tile
            key += (v * m)
            m *= self.base
        return key


    def q(self, afterstate, action):
        s = 0.0
        for tbl, tpl in zip(self.tables[action], self.n_tuples):
            k = self._encode(afterstate, tpl)
            s += tbl[k]
        return s / len(self.n_tuples)

    def update(self, afterstate, action, reward, next_afterstate, next_action, terminal=False):
        """
        afterstate: board after performing action (before random popup)
        next_afterstate: afterstate after taking next_action from next state (None if terminal)
        terminal: if next state is terminal
        """
        q_sa = self.q(afterstate, action)
        if terminal or next_afterstate is None:
            target = reward
        else:
            q_snext_anext = self.q(next_afterstate, next_action)
            target = reward + self.gamma * q_snext_anext
        delta = target - q_sa

        # apply update: add alpha * delta to each tuple entry corresponding to afterstate
        for i, tpl in enumerate(self.n_tuples):
            k = self._encode(afterstate, tpl)
            self.tables[action][i][k] += self.alpha * delta

    #policy: epsilon-greedy using Q(s,a) via afterstate simulation
    def act(self, game):
        legal = game.legal_moves()
        if not legal:
            return None
        if random.random() < self.epsilon:
            return random.choice(legal)

        best_val = float("-inf")
        best_a = None
        for a in legal:
            tmp = deepcopy(game)
            r = tmp.move(a)
            if r < 0:
                continue
            after = tmp.board
            val = r + self.q(after, a)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "n_tuples": self.n_tuples,
                "base": self.base,
                "tables": self.tables,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "max_tile": self.max_tile
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.n_tuples = payload["n_tuples"]
        self.base = payload["base"]
        self.tables = payload["tables"]
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.max_tile = payload.get("max_tile", self.max_tile)
