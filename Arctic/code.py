# a2c_minimal.py
import os, sys, time, csv, random
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env import Game2048

# -------- CONFIG ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 5000
GAMMA = 0.99
LR = 1e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
CHECKPOINT_EVERY = 200
SAVE_DIR = "a2c_min_checkpoints"
CSV_FILE = "a2c_min_log.csv"
# --------------------------

class ActorCritic(nn.Module):
    def __init__(self, in_dim=16, hidden=256, n_actions=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.trunk(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

def state_tensor(board):
    arr = np.array(board, dtype=np.float32) / 16.0
    return torch.tensor(arr, device=DEVICE).float()

def save_ckpt(path, model, opt, ep):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "episode": ep, "ts": time.time()}, path)

def load_ckpt(path, model, opt):
    data = torch.load(path, map_location=DEVICE)
    model.load_state_dict(data["model"])
    if "optimizer" in data and opt is not None:
        try:
            opt.load_state_dict(data["optimizer"])
        except Exception:
            pass
    return int(data.get("episode", 0))

def train(resume_ckpt=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model = ActorCritic().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)

    start_ep = 1
    if resume_ckpt and os.path.exists(resume_ckpt):
        loaded = load_ckpt(resume_ckpt, model, opt)
        start_ep = loaded + 1
        print(f"[A2C] Resuming from ep {loaded}")

    # CSV
    csv_mode = "a" if os.path.exists(CSV_FILE) and start_ep>1 else "w"
    f = open(CSV_FILE, csv_mode, newline="")
    writer = csv.writer(f)
    if csv_mode == "w":
        writer.writerow(["episode","score","steps","avg1000","policy_loss","value_loss","entropy","timestamp"])
        f.flush()

    scores_window = deque(maxlen=1000)
    pbar = tqdm(range(start_ep, EPISODES+1), desc="A2C")
    for ep in pbar:
        env = Game2048()
        state = tuple(env.board)
        done = False
        episode_transitions = []  # (s, a, logp, reward, value)
        ep_steps = 0
        ep_score = 0

        while not done:
            s_t = state_tensor(state).unsqueeze(0)
            logits, value = model(s_t)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            a = dist.sample().item()
            # ensure legal
            legal = env.legal_moves()
            if a not in legal:
                a = random.choice(legal)
                # recompute logp
                s_t = state_tensor(state).unsqueeze(0)
                logits, value = model(s_t)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)

            logp = dist.log_prob(torch.tensor(a, device=DEVICE)).item()
            value_scalar = value.item()

            before_empty = sum(1 for x in env.board if x == 0)
            r = env.move(a)
            if r < 0:
                reward = -20.0
                done = True
            else:
                after_empty = sum(1 for x in env.board if x == 0)
                reward = float(r) + 5.0*(after_empty - before_empty) - 1.0
                env.popup()
                done = len(env.legal_moves()) == 0

            next_state = tuple(env.board)
            episode_transitions.append((state, a, logp, reward, value_scalar))
            state = next_state
            ep_steps += 1
            ep_score += reward

            if done:
                break
            if ep_steps > 2000:
                break

        # compute returns and advantages (simple)
        returns = []
        R = 0.0
        for _,_,_,r,_ in reversed(episode_transitions):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32)
        values = torch.tensor([t[4] for t in episode_transitions], device=DEVICE, dtype=torch.float32)
        advantages = returns - values

        # convert to tensors for update
        states = torch.stack([state_tensor(t[0]) for t in episode_transitions]).to(DEVICE)
        actions = torch.tensor([t[1] for t in episode_transitions], device=DEVICE).long()
        old_logps = torch.tensor([t[2] for t in episode_transitions], device=DEVICE).float()

        # compute losses
        logits, value_preds = model(states)
        dists = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
        logps = dists.log_prob(actions)
        entropy = dists.entropy().mean()

        policy_loss = -(logps * advantages.detach()).mean()
        value_loss = (returns - value_preds).pow(2).mean()

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()

        scores_window.append(int(env.score))
        avg1000 = float(sum(scores_window)/len(scores_window))
        writer.writerow([ep, int(env.score), ep_steps, avg1000, float(policy_loss.item()), float(value_loss.item()), float(entropy.item()), int(time.time())])
        f.flush()

        if ep % CHECKPOINT_EVERY == 0:
            save_ckpt(os.path.join(SAVE_DIR, f"a2c_ep{ep:04d}.pt"), model, opt, ep)

        pbar.set_postfix({"ep":ep,"last_score":int(env.score),"avg1000":f"{avg1000:.1f}"})

    # final save
    save_ckpt(os.path.join(SAVE_DIR,"a2c_final.pt"), model, opt, EPISODES)
    torch.save(model.state_dict(), "a2c_model_final.pt")
    f.close()
    print("A2C training finished.")

if __name__ == "__main__":
    resume = sys.argv[1] if len(sys.argv)>1 else None
    train(resume)