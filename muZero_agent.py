# muzero_agent.py
import torch
import torch.nn as nn
import numpy as np
import random

###############################################################################
# MuZero Mini — for 2048
###############################################################################

class Representation(nn.Module):
    """ Encodes board into hidden state. """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


class Dynamics(nn.Module):
    """ Predicts next hidden state + reward given (state, action) """
    def __init__(self, hidden=128, n_actions=4):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(hidden + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.reward = nn.Linear(hidden + n_actions, 1)

    def forward(self, h, a_onehot):
        x = torch.cat([h, a_onehot], dim=1)
        next_state = self.state(x)
        reward = self.reward(x)
        return next_state, reward


class Prediction(nn.Module):
    """ Predicts policy logits + value from hidden state """
    def __init__(self, hidden=128, n_actions=4):
        super().__init__()
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, h):
        return self.policy(h), self.value(h)


###############################################################################
# MuZero Agent (inference only)
###############################################################################
def board_to_tensor(board, device):
    arr = np.array(board, np.float32) / 16.0
    return torch.tensor(arr, device=device).float().unsqueeze(0)


class MuZeroAgent:
    def __init__(self, path="muzero.pt", device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        hidden = 128
        self.repr = Representation(hidden).to(self.device)
        self.dyn = Dynamics(hidden).to(self.device)
        self.pred = Prediction(hidden).to(self.device)

        # ---- load checkpoint ----
        data = torch.load(path, map_location=self.device)
        self.repr.load_state_dict(data["repr"])
        self.dyn.load_state_dict(data["dyn"])
        self.pred.load_state_dict(data["pred"])

        self.repr.eval()
        self.dyn.eval()
        self.pred.eval()

    def act(self, game):
        legal = game.legal_moves()
        if not legal:
            return None

        board = tuple(game.board)
        s = board_to_tensor(board, self.device)

        # encode
        with torch.no_grad():
            h = self.repr(s)

        # simple 1-step lookahead MuZero MCTS
        scores = {}

        for a in legal:
            a_onehot = torch.zeros((1, 4), device=self.device)
            a_onehot[0, a] = 1

            with torch.no_grad():
                next_h, reward = self.dyn(h, a_onehot)
                policy_logits, value = self.pred(next_h)

            scores[a] = reward.item() + value.item()

        best = max(scores, key=scores.get)
        return best