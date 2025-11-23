# plot_training.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# -----------------------------------------------
# Moving average helper
# -----------------------------------------------
def smooth(series, window=50):
    if len(series) < window:
        return series
    return np.convolve(series, np.ones(window)/window, mode="valid")

# -----------------------------------------------
# Main Plotter
# -----------------------------------------------
def plot_csv(csv_path, window=50):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)

    print("\nColumns found:", df.columns.tolist())

    # Detect columns automatically
    score_col     = "score"       if "score" in df.columns     else None
    avg_col       = "avg100"      if "avg100" in df.columns    else None
    eps_col       = "epsilon"     if "epsilon" in df.columns   else None
    pol_col       = "policy_loss" if "policy_loss" in df.columns else None
    val_col       = "value_loss"  if "value_loss" in df.columns  else None
    rew_col       = "reward_loss" if "reward_loss" in df.columns else None

    # -----------------------------------------------
    # 1. Score curve
    # -----------------------------------------------
    if score_col:
        plt.figure(figsize=(10,6))
        plt.plot(df["episode"], df[score_col], label="Score", alpha=0.4)
        plt.plot(df["episode"].iloc[:len(smooth(df[score_col], window))],
                 smooth(df[score_col], window), label=f"Smoothed ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Score vs Episode")
        plt.grid()
        plt.legend()
        plt.savefig("score_curve.png")
        plt.close()
        print("Saved: score_curve.png")

    # -----------------------------------------------
    # 2. Average window curve
    # -----------------------------------------------
    if avg_col:
        plt.figure(figsize=(10,6))
        plt.plot(df["episode"], df[avg_col], label="Average Last-Window")
        plt.xlabel("Episode")
        plt.ylabel("Avg Window Score")
        plt.title("Moving Average Score")
        plt.grid()
        plt.legend()
        plt.savefig("avg_curve.png")
        plt.close()
        print("Saved: avg_curve.png")

    # -----------------------------------------------
    # 3. Losses (MuZero or PPO)
    # -----------------------------------------------
    if pol_col or val_col or rew_col:
        plt.figure(figsize=(10,6))
        if pol_col:
            plt.plot(df["episode"], df[pol_col], label="Policy Loss")
        if val_col:
            plt.plot(df["episode"], df[val_col], label="Value Loss")
        if rew_col:
            plt.plot(df["episode"], df[rew_col], label="Reward Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.grid()
        plt.legend()
        plt.savefig("loss_curves.png")
        plt.close()
        print("Saved: loss_curves.png")

    # -----------------------------------------------
    # 4. Epsilon curve (for DQN)
    # -----------------------------------------------
    if eps_col:
        plt.figure(figsize=(10,6))
        plt.plot(df["episode"], df[eps_col], label="Epsilon")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Exploration Decay")
        plt.grid()
        plt.legend()
        plt.savefig("epsilon_curve.png")
        plt.close()
        print("Saved: epsilon_curve.png")

    print("\nAll graphs saved successfully!")


# -----------------------------------------------
# Run from command line
# -----------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <csv_file>")
        sys.exit(0)

    plot_csv(sys.argv[1])