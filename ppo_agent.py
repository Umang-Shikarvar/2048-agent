# ppo_agent.py
import torch
import torch.nn as nn
import numpy as np
import random

class ActorCritic(nn.Module):
    def __init__(self, input_dim=16, hidden=256, n_actions=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        logits = self.policy(h)
        value = self.value(h)
        return logits, value

def board_to_tensor(board, device):
    arr = np.array(board, dtype=np.float32) / 16.0
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

class PPOAgent:
    def __init__(self, path="ppo_model_only.pt", device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = ActorCritic().to(self.device)
        data = torch.load(path, map_location=self.device)

        if isinstance(data, dict) and "model" in data:
            self.model.load_state_dict(data["model"])
        else:
            try:
                self.model.load_state_dict(data)
            except:
                self.model = data.to(self.device)

        self.model.eval()

    def act(self, game):
        legal = game.legal_moves()
        if not legal:
            return None

        s = tuple(game.board)
        x = board_to_tensor(s, self.device)

        with torch.no_grad():
            logits, _ = self.model(x)
            logits = logits.squeeze(0).cpu().numpy()

        masked = np.full(4, -1e9, np.float32)
        for a in legal:
            masked[a] = logits[a]

        a = int(np.argmax(masked))
        if a not in legal:
            a = int(random.choice(legal))

        return a