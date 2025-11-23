# muzero_lite.py
"""
MuZero-Lite for 2048 (fast & small)
- Small networks (fast on T4)
- MCTS with a small number of simulations (default 5)
- Checkpoints + CSV logging
- Resume support
- Designed to finish reasonably fast on a T4 (tuned defaults)
"""

import os
import sys
import time
import math
import random
import json
import csv
from collections import deque, namedtuple, defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Use your env.py in the same folder (example path: /mnt/zone/C/LHM_animal/QNDeep/env.py)
from env import Game2048

# -------------------------
# Config (tuned for T4, ~1 hour)
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODE_GOAL = 5000            # total training episodes (set lower so it finishes fast), adjust as needed
CHECKPOINT_EVERY = 100
SAVE_DIR = "muzero_checkpoints"
CSV_FILE = "muzero_log.csv"
STATE_JSON = "muzero_state.json"

# MuZero hyperparams (fast)
MCTS_SIMS = 5                  # you wanted quick runs -> 5 sims per move
C_PUCT = 1.0
TD_STEPS = 5                   # unroll length for training (how many steps predicted)
REPLAY_CAPACITY = 20000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
TRAIN_UPDATES_PER_EP = 8       # updates per episode (small to keep it quick)
NUM_EPOCHS = 1                 # epochs per update over sampled batches
KL_COEFF = 1.0

# Network sizes (small)
HIDDEN = 128
LATENT = 64
ACTION_SIZE = 4                # up,right,down,left

# Keep reproducible-ish
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Transition for replay
SelfPlayTransition = namedtuple("SelfPlayTransition",
                                ["obs", "root_policy", "value_target", "reward_target", "to_play"])

# -------------------------
# Small networks for MuZero-lite
# -------------------------
class RepresentationNet(nn.Module):
    def __init__(self, in_dim=16, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)  # latent vector

class DynamicsNet(nn.Module):
    def __init__(self, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + ACTION_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU()
        )
        # predict reward
        self.reward_head = nn.Sequential(nn.Linear(latent, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, latent, action_onehot):
        inp = torch.cat([latent, action_onehot], dim=-1)
        next_latent = self.net(inp)
        reward = self.reward_head(next_latent).squeeze(-1)
        return next_latent, reward

class PredictionNet(nn.Module):
    def __init__(self, latent=LATENT, hidden=HIDDEN, action_size=ACTION_SIZE):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(latent, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, action_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, latent):
        logits = self.policy_head(latent)             # raw logits for policy
        value = self.value_head(latent).squeeze(-1)   # scalar
        return logits, value

# -------------------------
# Utilities
# -------------------------
def board_to_tensor(board):
    arr = np.array(board, dtype=np.float32) / 16.0
    return torch.tensor(arr, device=DEVICE).float()

def one_hot_action(a):
    v = torch.zeros(ACTION_SIZE, device=DEVICE, dtype=torch.float32)
    v[a] = 1.0
    return v

def softmax_logits_to_policy(logits):
    probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()

# -------------------------
# Simple MCTS node
# -------------------------
class MCTSNode:
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}   # action -> child node
        self.prior = 0.0
        self.reward = 0.0

    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

# -------------------------
# MuZero Agent: self-play with MCTS
# -------------------------
class MuZeroAgent:
    def __init__(self, repr_net, dyn_net, pred_net, device=DEVICE):
        self.repr = repr_net
        self.dyn = dyn_net
        self.pred = pred_net
        self.device = device

    def run_mcts(self, root_state, root_latent, legal_moves, sims=MCTS_SIMS):
        root = MCTSNode()
        # initial policy & value from prediction
        with torch.no_grad():
            logits, value = self.pred(root_latent.unsqueeze(0))
            logits = logits.squeeze(0)
            value = value.item()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # mask illegal moves by zeroing their prior then renormalize
        priors = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a in legal_moves:
            priors[a] = max(probs[a], 1e-6)
        if priors.sum() == 0:
            priors[legal_moves] = 1.0 / len(legal_moves)
        else:
            priors = priors / priors.sum()
        for a in range(ACTION_SIZE):
            child = MCTSNode()
            child.prior = float(priors[a])
            root.children[a] = child

        # MCTS sims
        for _ in range(sims):
            node = root
            latent = root_latent.clone()
            search_path = [node]
            actions_taken = []

            # selection
            while True:
                # pick best UCB among children
                best_score = -float("inf")
                best_action = None
                for a, child in node.children.items():
                    Q = child.value()
                    U = C_PUCT * child.prior * math.sqrt(node.visit_count + 1) / (1 + child.visit_count)
                    score = Q + U
                    if score > best_score:
                        best_score = score
                        best_action = a
                # move to child
                node = node.children[best_action]
                actions_taken.append(best_action)
                # expand if leaf (no children under node)
                if len(node.children) == 0:
                    break

            # expansion: simulate dynamics from latent through actions_taken to reach leaf
            for a in actions_taken:
                a_oh = one_hot_action(a).unsqueeze(0)  # (1, A)
                latent, reward = self.dyn(latent.unsqueeze(0), a_oh)
                latent = latent.squeeze(0)
                node.reward = float(reward.item())

            # evaluate leaf with prediction network
            with torch.no_grad():
                logits, value = self.pred(latent.unsqueeze(0))
                logits = logits.squeeze(0)
                value = float(value.item())

            # backpropagate value
            # use reward of leaf + value as target
            target_value = value
            # backpropagation through path
            # For simplicity, assign same target to all nodes visited
            for visited_node in reversed(search_path + [node]):
                visited_node.visit_count += 1
                visited_node.value_sum += target_value

        # extract policy from root visit counts
        visit_counts = np.array([root.children[a].visit_count for a in range(ACTION_SIZE)], dtype=np.float32)
        if visit_counts.sum() == 0:
            # fallback uniform over legal moves
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[legal_moves] = 1.0 / len(legal_moves)
        else:
            policy = visit_counts / visit_counts.sum()
        return policy, root

# -------------------------
# Replay buffer (simple ring)
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Training functions
# -------------------------
def build_targets_from_selfplay(trajectories):
    """Given list of self-play trajectories, build training tuples:
       For each position, get value target for next TD_STEPS and reward sequence for TD_STEPS."""
    # Each trajectory: list of (obs, root_policy (array), ) from self-play until terminal or cutoff
    training_examples = []
    for traj in trajectories:
        # traj: list of dicts with keys obs, root_policy, rewards (sequence), done flags
        n = len(traj)
        for i in range(n):
            # value target: discounted sum over next TD_STEPS of rewards (bootstrap with model value=0)
            value = 0.0
            discount = 1.0
            for k in range(TD_STEPS):
                if i + k < n:
                    value += discount * traj[i + k]["reward"]
                    discount *= 0.99
                else:
                    break
            # reward target for the first step
            reward_target = traj[i]["reward"]
            # store
            training_examples.append({
                "obs": traj[i]["obs"],
                "policy": traj[i]["root_policy"],
                "value": value,
                "reward": reward_target
            })
    return training_examples

def train_step(policy_model_tuple, optimizer, batch):
    """Train on a batch of examples (batch is list of dicts with obs, policy, value, reward)"""
    repr_net, dyn_net, pred_net = policy_model_tuple
    repr_net.train(); dyn_net.train(); pred_net.train()

    # build tensors
    obs = torch.stack([board_to_tensor(x["obs"]) for x in batch]).to(DEVICE)  # (B,16)
    target_policies = torch.tensor([x["policy"] for x in batch], dtype=torch.float32, device=DEVICE)
    target_values = torch.tensor([x["value"] for x in batch], dtype=torch.float32, device=DEVICE)
    target_rewards = torch.tensor([x["reward"] for x in batch], dtype=torch.float32, device=DEVICE)

    # representation
    latent = repr_net(obs)  # (B, latent)

    # prediction from latent (policy + value)
    logits, values = pred_net(latent)
    # policy loss (cross-entropy with target policy)
    logp = F.log_softmax(logits, dim=-1)
    policy_loss = - (target_policies * logp).sum(dim=-1).mean()

    # value loss
    value_loss = F.mse_loss(values, target_values)

    # dynamics/reward loss: for simplicity we only predict reward from a zero-action input (or use action 0)
    # We keep this simple: predict reward as a small head from latent; no unroll here to keep fast
    # Use target_rewards as supervision for a reward head approximated inside dynamics (we use zero-action)
    # prepare zero action
    action_zero = torch.zeros(len(batch), ACTION_SIZE, device=DEVICE)
    next_latent, reward_pred = dyn_net(latent, action_zero)
    reward_loss = F.mse_loss(reward_pred, target_rewards)

    total_loss = policy_loss + value_loss + reward_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(repr_net.parameters()) + list(dyn_net.parameters()) + list(pred_net.parameters()), max_norm=1.0)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item()), float(reward_loss.item())

# -------------------------
# Main training loop
# -------------------------
def train_muzero(resume_path=None):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # networks
    repr_net = RepresentationNet().to(DEVICE)
    dyn_net = DynamicsNet().to(DEVICE)
    pred_net = PredictionNet().to(DEVICE)

    # optimizer (joint)
    params = list(repr_net.parameters()) + list(dyn_net.parameters()) + list(pred_net.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    # replay buffer
    replay = ReplayBuffer(REPLAY_CAPACITY)

    # resume if provided
    completed = 0
    if resume_path and os.path.exists(resume_path):
        data = torch.load(resume_path, map_location=DEVICE)
        if "model_state" in data:
            repr_net.load_state_dict(data["model_state"]["repr"])
            dyn_net.load_state_dict(data["model_state"]["dyn"])
            pred_net.load_state_dict(data["model_state"]["pred"])
            if "optimizer_state" in data:
                try:
                    optimizer.load_state_dict(data["optimizer_state"])
                except Exception:
                    pass
            completed = int(data.get("episodes", 0))
            print(f"[MuZero] Resumed from {resume_path}, completed episodes = {completed}")
        else:
            print("[MuZero] Unknown checkpoint format, starting fresh.")

    # logging CSV
    csv_mode = "a" if os.path.exists(CSV_FILE) and completed > 0 else "w"
    csv_f = open(CSV_FILE, csv_mode, newline="")
    csv_writer = csv.writer(csv_f)
    if csv_mode == "w":
        csv_writer.writerow(["episode","score","avg100","policy_loss","value_loss","reward_loss","timestamp"])
        csv_f.flush()

    scores_window = deque(maxlen=100)
    pbar = tqdm(total=EPISODE_GOAL, desc="MuZero-lite (episodes)")
    pbar.update(completed)

    agent = MuZeroAgent(repr_net, dyn_net, pred_net, device=DEVICE)

    # main loop: self-play episodes until completed reaches target
    while completed < EPISODE_GOAL:
        env = Game2048()
        obs = tuple(env.board)
        traj = []  # self-play trajectory records
        steps = 0
        # play until done
        while True:
            # get latent for root
            with torch.no_grad():
                root_latent = repr_net(board_to_tensor(obs).unsqueeze(0)).squeeze(0)
            legal = env.legal_moves()
            if not legal:
                break
            # run MCTS to get policy
            root_policy, _ = agent.run_mcts(obs, root_latent, legal, sims=MCTS_SIMS)
            # sample action from root_policy but restrict to legal moves
            # normalize legal part
            probs = np.array(root_policy, dtype=np.float32)
            probs_masked = np.zeros_like(probs)
            probs_masked[legal] = probs[legal]
            if probs_masked.sum() == 0:
                probs_masked[legal] = 1.0 / len(legal)
            probs_masked = probs_masked / probs_masked.sum()
            action = int(np.random.choice(np.arange(ACTION_SIZE), p=probs_masked))
            # step environment
            before_empty = sum(1 for x in env.board if x == 0)
            r = env.move(action)
            if r < 0:
                reward = -20.0
                done = True
            else:
                after_empty = sum(1 for x in env.board if x == 0)
                reward = float(r) + 5.0 * (after_empty - before_empty) - 1.0
                env.popup()
                done = len(env.legal_moves()) == 0
            next_obs = tuple(env.board)
            traj.append({"obs": obs, "root_policy": root_policy, "reward": reward})
            obs = next_obs
            steps += 1
            if done:
                break

            # safety to avoid infinite loops
            if steps > 1000:
                break

        # episode finished
        score = env.score
        completed += 1
        scores_window.append(score)
        avg100 = float(sum(scores_window)/len(scores_window))
        # push trajectory frames into replay buffer as simplified training examples
        examples = build_targets_from_selfplay([traj])
        for ex in examples:
            replay.push(ex)

        # training updates (small number per episode)
        policy_loss_avg = value_loss_avg = reward_loss_avg = 0.0
        if len(replay) >= BATCH_SIZE:
            for _ in range(TRAIN_UPDATES_PER_EP):
                batch = replay.sample(BATCH_SIZE)
                pl, vl, rl = train_step((repr_net, dyn_net, pred_net), optimizer, batch)
                policy_loss_avg += pl; value_loss_avg += vl; reward_loss_avg += rl
            policy_loss_avg /= max(1, TRAIN_UPDATES_PER_EP)
            value_loss_avg /= max(1, TRAIN_UPDATES_PER_EP)
            reward_loss_avg /= max(1, TRAIN_UPDATES_PER_EP)

        csv_writer.writerow([completed, score, avg100, policy_loss_avg, value_loss_avg, reward_loss_avg, int(time.time())])
        csv_f.flush()
        # save a lightweight state JSON
        with open(STATE_JSON, "w") as jf:
            json.dump({"episode": completed, "score": score, "avg100": avg100}, jf)

        # checkpoint
        if completed % CHECKPOINT_EVERY == 0:
            ckpt = {
                "model_state": {
                    "repr": repr_net.state_dict(),
                    "dyn": dyn_net.state_dict(),
                    "pred": pred_net.state_dict()
                },
                "optimizer_state": optimizer.state_dict(),
                "episodes": completed,
                "ts": time.time()
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, f"muzero_ep{completed:04d}.pt"))

        pbar.update(1)

    pbar.close()
    csv_f.close()

    # final save
    final_ckpt = os.path.join(SAVE_DIR, "muzero_final.pt")
    ckpt = {
        "model_state": {
            "repr": repr_net.state_dict(),
            "dyn": dyn_net.state_dict(),
            "pred": pred_net.state_dict()
        },
        "optimizer_state": optimizer.state_dict(),
        "episodes": completed,
        "ts": time.time()
    }
    torch.save(ckpt, final_ckpt)
    print("[MuZero] Training finished. Episodes:", completed)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train_muzero(resume)