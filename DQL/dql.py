# dqn_minimal.py
import os, sys, time, random, csv, pickle
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from env import Game2048

# -------- CONFIG (tweak as needed) ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 5000
BATCH_SIZE = 512
BUFFER_CAPACITY = 200000
LR = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_MIN = 0.1
EPS_DECAY = 0.9993
TARGET_UPDATE = 1000
CHECKPOINT_EVERY = 200
SAVE_DIR = "dqn_min_checkpoints"
CSV_FILE = "dqn_min_log.csv"
# --------------------------------------------

class DQNNet(nn.Module):
    def __init__(self, in_dim=16, hidden=256, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def state_tensor(board):
    arr = np.array(board, dtype=np.float32) / 16.0
    return torch.tensor(arr, device=DEVICE).float()

def save_ckpt(path, policy, target, opt, ep):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "target": target.state_dict(),
        "optimizer": opt.state_dict(),
        "episode": ep,
        "ts": time.time()
    }, path)

def load_ckpt(path, policy, target, opt):
    data = torch.load(path, map_location=DEVICE)
    policy.load_state_dict(data["policy"])
    target.load_state_dict(data["target"])
    if opt is not None and "optimizer" in data:
        try:
            opt.load_state_dict(data["optimizer"])
        except Exception:
            pass
    return int(data.get("episode", 0))

def train(resume_ckpt=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    policy = DQNNet().to(DEVICE)
    target = DQNNet().to(DEVICE)
    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=LR)
    replay = deque(maxlen=BUFFER_CAPACITY)

    start_ep = 1
    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            ep_loaded = load_ckpt(resume_ckpt, policy, target, opt)
            start_ep = ep_loaded + 1
            print(f"[DQN] Resuming from ep {ep_loaded}")
        else:
            print("[DQN] Resume checkpoint not found; starting fresh.")

    eps = EPS_START
    steps = 0

    # CSV logging
    csv_mode = "a" if os.path.exists(CSV_FILE) and start_ep > 1 else "w"
    f = open(CSV_FILE, csv_mode, newline="")
    writer = csv.writer(f)
    if csv_mode == "w":
        writer.writerow(["episode","score","avg1000","epsilon","loss","ts"])
        f.flush()

    scores_window = deque(maxlen=1000)
    all_scores = []

    pbar = tqdm(range(start_ep, EPISODES+1), desc="DQN")
    for ep in pbar:
        env = Game2048()
        done = False
        ep_loss = 0.0
        while not done:
            s = tuple(env.board)
            legal = env.legal_moves()
            if not legal:
                break
            if random.random() < eps:
                a = random.choice(legal)
            else:
                with torch.no_grad():
                    q = policy(state_tensor(s)).cpu().numpy()
                a = int(max(legal, key=lambda x: q[x]))

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

            s2 = tuple(env.board)
            replay.append((s,a,reward,s2,done))
            steps += 1

            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)
                S,A,R,S2,D = zip(*batch)
                S = torch.stack([state_tensor(x) for x in S]).to(DEVICE)
                S2 = torch.stack([state_tensor(x) for x in S2]).to(DEVICE)
                A = torch.tensor(A, device=DEVICE).long()
                R = torch.tensor(R, device=DEVICE).float()
                D = torch.tensor(D, device=DEVICE).float()

                qvals = policy(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(S2).max(1)[0]
                    target_q = R + (1.0 - D) * GAMMA * next_q
                loss = nn.MSELoss()(qvals, target_q)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                opt.step()
                ep_loss = loss.item()

            if steps % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

        scores_window.append(env.score)
        all_scores.append(env.score)
        avg1000 = float(sum(scores_window)/len(scores_window))

        writer.writerow([ep, env.score, avg1000, float(eps), ep_loss, int(time.time())])
        f.flush()

        if ep % CHECKPOINT_EVERY == 0:
            save_ckpt(os.path.join(SAVE_DIR, f"dqn_ep{ep:04d}.pt"), policy, target, opt, ep)

        eps = max(eps * EPS_DECAY, EPS_MIN)
        pbar.set_postfix({"ep":ep,"eps":f"{eps:.4f}","last":env.score,"avg1000":f"{avg1000:.1f}"})

    # final saves
    save_ckpt(os.path.join(SAVE_DIR,"dqn_final.pt"), policy, target, opt, EPISODES)
    torch.save(policy.state_dict(), "dqn_policy_final.pt")
    f.close()
    print("DQN training finished.")

if __name__ == "__main__":
    resume = sys.argv[1] if len(sys.argv)>1 else None
    train(resume)