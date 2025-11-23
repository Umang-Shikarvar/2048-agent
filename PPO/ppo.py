# ppo.py
import os
import sys
import time
import csv
import json
import random
from collections import deque, namedtuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env import Game2048

# -------------------------
# Config (tweak here)
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODE_GOAL = 5000            # total episodes you want to collect (completed games)
ROLLOUT_STEPS = 2048           # timesteps per rollout (increase for stability)
UPDATE_EPOCHS = 8              # epochs to update per rollout
MINI_BATCH_SIZE = 256          # minibatch size for PPO updates
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 2.5e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

CHECKPOINT_EVERY = 200         # save checkpoint every N completed episodes
SAVE_DIR = "ppo_checkpoints"
CSV_FILE = "ppo_log.csv"
JSON_STATE = "ppo_state.json"

# -------------------------
# Network (shared actor-critic)
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim=16, hidden=256, n_actions=4):
        super().__init__()
        # common trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # actor head
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, n_actions)
        )
        # critic head
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        # x: tensor shape (..., 16)
        h = self.trunk(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

# -------------------------
# Utilities
# -------------------------
def state_to_tensor(board):
    arr = np.array(board, dtype=np.float32) / 16.0
    return torch.tensor(arr, device=DEVICE).float()

Transition = namedtuple("Transition", ["state", "action", "logp", "reward", "mask", "value"])

def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    rewards: list of rewards
    masks: list of masks (1 - done)
    values: list of values (floats)
    next_value: scalar tensor
    returns:
        - returns (list)
        - advantages (list)
    """
    values = values + [next_value.item()]
    gae = 0.0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])
    return returns, advantages

def make_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)

def save_checkpoint(path, model, optimizer, completed_episodes):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "episodes": completed_episodes,
        "ts": time.time()
    }
    torch.save(payload, path)

def load_checkpoint(path, model, optimizer):
    d = torch.load(path, map_location=DEVICE)
    if "model_state" in d:
        model.load_state_dict(d["model_state"])
        if optimizer is not None and "optimizer_state" in d:
            try:
                optimizer.load_state_dict(d["optimizer_state"])
            except Exception:
                pass
        return int(d.get("episodes", 0))
    else:
        # fallback for pure state dict
        model.load_state_dict(d)
        return 0

# -------------------------
# PPO trainer
# -------------------------
def train_ppo(resume_ckpt=None):
    make_save_dir()
    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # logging
    csv_mode = "a" if os.path.exists(CSV_FILE) and resume_ckpt else "w"
    csv_f = open(CSV_FILE, csv_mode, newline="")
    csv_writer = csv.writer(csv_f)
    if csv_mode == "w":
        csv_writer.writerow(["episode","score","steps","avg1000","policy_loss","value_loss","entropy","timestamp"])
        csv_f.flush()

    completed_episodes = 0
    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            print("Resuming from:", resume_ckpt)
            completed_episodes = load_checkpoint(resume_ckpt, model, optimizer)
            print("Resumed completed_episodes =", completed_episodes)
        else:
            print("Resume checkpoint not found, starting fresh.")

    scores_window = deque(maxlen=1000)
    learning_curve = []

    env = Game2048()  # we'll use a single environment but collect rollouts across episodes
    state = tuple(env.board)
    episode_step = 0
    episode_score = 0

    pbar = tqdm(total=EPISODE_GOAL, desc="PPO (episodes)")
    pbar.update(completed_episodes)

    # main loop: keep collecting until completed_episodes reaches goal
    while completed_episodes < EPISODE_GOAL:
        # ==== Collect rollout ====
        transitions = []
        steps_collected = 0
        while steps_collected < ROLLOUT_STEPS:
            s_tensor = state_to_tensor(state).unsqueeze(0)  # (1,16)
            with torch.no_grad():
                logits, value = model(s_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                value_scalar = value.item()

            legal = env.legal_moves()
            if action not in legal:
                # safe fallback: choose legal randomly (keeps training stable)
                action = random.choice(legal)
                # recompute logp and value properly for chosen action
                with torch.no_grad():
                    logits, value = model(state_to_tensor(state).unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                    value_scalar = value.item()

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

            next_state = tuple(env.board)
            mask = 0.0 if done else 1.0

            transitions.append(Transition(state, action, logp, reward, mask, value_scalar))

            state = next_state
            steps_collected += 1
            episode_step += 1
            episode_score += 0 if reward is None else (reward if isinstance(reward, (int, float)) else 0)

            if done:
                # episode finished: log it
                completed_episodes += 1
                scores_window.append(env.score)
                learning_curve.append(sum(scores_window)/len(scores_window) if len(scores_window)>0 else env.score)
                csv_writer.writerow([completed_episodes, env.score, episode_step, (sum(scores_window)/len(scores_window)), 0.0, 0.0, 0.0, int(time.time())])
                csv_f.flush()
                # reset env
                env = Game2048()
                state = tuple(env.board)
                episode_step = 0
                episode_score = 0
                pbar.update(1)

                # checkpoint per CHECKPOINT_EVERY completed episodes
                if completed_episodes % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(SAVE_DIR, f"ppo_ep{completed_episodes:04d}.pt")
                    save_checkpoint(ckpt_path, model, optimizer, completed_episodes)

                # stop collecting if we've reached goal
                if completed_episodes >= EPISODE_GOAL:
                    break

        # If no transitions collected (rare), continue
        if len(transitions) == 0:
            continue

        # ==== Prepare training batch ====
        # We'll compute GAE and returns
        # Extract arrays
        states = [t.state for t in transitions]
        actions = [t.action for t in transitions]
        old_logps = [t.logp for t in transitions]
        rewards = [t.reward for t in transitions]
        masks = [t.mask for t in transitions]
        values = [t.value for t in transitions]

        # compute next_value for the last state
        with torch.no_grad():
            last_state = state_to_tensor(state).unsqueeze(0)
            _, next_value = model(last_state)
            next_value = next_value.squeeze(0)

        returns, advantages = compute_gae(next_value, rewards, masks, values, GAMMA, GAE_LAMBDA)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert arrays to tensors for batching
        S_tensor = torch.stack([state_to_tensor(s) for s in states]).to(DEVICE)
        A_tensor = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        old_logp_tensor = torch.tensor(old_logps, dtype=torch.float32, device=DEVICE)

        # ==== PPO update (multiple epochs) ====
        dataset_size = S_tensor.size(0)
        for epoch in range(UPDATE_EPOCHS):
            # iterate over minibatches
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                mb_idx = idxs[start:start + MINI_BATCH_SIZE]
                mb_idx = torch.tensor(mb_idx, dtype=torch.long, device=DEVICE)

                mb_states = S_tensor[mb_idx]
                mb_actions = A_tensor[mb_idx]
                mb_old_logp = old_logp_tensor[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                logits, values_pred = model(mb_states)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                mb_new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(mb_new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_ret - values_pred).pow(2).mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # After update, log summary of this rollout (we will log per-episode rows already)
        # Save a general JSON checkpoint of last state
        with open(JSON_STATE, "w") as jf:
            json.dump({
                "completed_episodes": completed_episodes,
                "last_rollout_size": len(transitions),
                "timestamp": int(time.time())
            }, jf)

    pbar.close()
    csv_f.close()

    # final save
    final_ckpt = os.path.join(SAVE_DIR, "ppo_final.pt")
    save_checkpoint(final_ckpt, model, optimizer, completed_episodes)

    # Optionally save the model-only weights for inference
    torch.save(model.state_dict(), "ppo_model_only.pt")
    print("PPO training finished. Completed episodes:", completed_episodes)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train_ppo(resume)