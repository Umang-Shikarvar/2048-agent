# ppo_agent.py
import torch
import torch.nn as nn
import numpy as np
import random


# SAME MODEL AS TRAINING
class ActorCritic(nn.Module):
    def __init__(self, input_dim=16, hidden=256, n_actions=4):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

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

        # CASE 1 — Saved checkpoint (model_state inside dict)
        if isinstance(data, dict) and "model_state" in data:
            print("[PPOAgent] Loading model_state from checkpoint")
            self.model.load_state_dict(data["model_state"], strict=True)

        # CASE 2 — model-only (state_dict)
        else:
            print("[PPOAgent] Loading raw state_dict")
            self.model.load_state_dict(data, strict=True)

        self.model.eval()

    def act(self, game):
        legal = game.legal_moves()
        if not legal:
            return None

        s = tuple(game.board)
        x = board_to_tensor(s, self.device)

        with torch.no_grad():
            logits, _ = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)

        # Mask illegal moves
        masked = np.full(4, -1e9, np.float32)
        for a in legal:
            masked[a] = probs[a]

        action = int(np.argmax(masked))

        # Final safety
        if action not in legal:
            action = int(random.choice(legal))

        return action