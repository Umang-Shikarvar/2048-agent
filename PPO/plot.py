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

    # -----------------------------------------------
    # Detect score column
    # -----------------------------------------------
    if "score" not in df.columns:
        print("Error: CSV does not contain 'score' column.")
        return
    score_col = "score"

    # -----------------------------------------------
    # Detect or **create avg100** (never use avg1000)
    # -----------------------------------------------
    if "avg100" in df.columns:
        avg_col = "avg100"
        print("Using existing column: avg100")
    else:
        print("avg100 not found — computing avg100 from scores...")
        df["avg100"] = df["score"].rolling(window=100).mean()
        avg_col = "avg100"

    # -----------------------------------------------
    # Detect optional DQN fields
    # -----------------------------------------------
    eps_col = "epsilon" if "epsilon" in df.columns else None

    # -----------------------------------------------
    # Detect optional loss fields (PPO, AC, MuZero)
    # -----------------------------------------------
    pol_col = "policy_loss" if "policy_loss" in df.columns else None
    val_col = "value_loss"  if "value_loss" in df.columns else None
    rew_col = "reward_loss" if "reward_loss" in df.columns else None

    # -----------------------------------------------
    # 1. Score curve
    # -----------------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(df["episode"], df["score"], label="Score", alpha=0.4)

    smoothed = smooth(df["score"], window)
    plt.plot(df["episode"].iloc[:len(smoothed)], smoothed,
             label=f"Smoothed ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score vs Episode")
    plt.grid()
    plt.legend()
    plt.savefig("score_curve.png")
    plt.close()
    print("Saved: score_curve.png")

    # -----------------------------------------------
    # 2. avg100 curve (computed or existing)
    # -----------------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(df["episode"], df[avg_col], label="avg100 (moving avg)", color="green")
    plt.xlabel("Episode")
    plt.ylabel("avg100 score")
    plt.title("avg100 (Moving Average of Scores)")
    plt.grid()
    plt.legend()
    plt.savefig("avg_curve.png")
    plt.close()
    print("Saved: avg_curve.png")

    # -----------------------------------------------
    # 3. Loss curves (if exist)
    # -----------------------------------------------
    if pol_col or val_col or rew_col:
        plt.figure(figsize=(10,6))
        if pol_col: plt.plot(df["episode"], df[pol_col], label="Policy Loss")
        if val_col: plt.plot(df["episode"], df[val_col], label="Value Loss")
        if rew_col: plt.plot(df["episode"], df[rew_col], label="Reward Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Loss curves")
        plt.grid()
        plt.legend()
        plt.savefig("loss_curves.png")
        plt.close()
        print("Saved: loss_curves.png")

    # -----------------------------------------------
    # 4. Epsilon curve (if exists)
    # -----------------------------------------------
    if eps_col:
        plt.figure(figsize=(10,6))
        plt.plot(df["episode"], df[eps_col])
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Exploration epsilon")
        plt.grid()
        plt.savefig("epsilon_curve.png")
        plt.close()
        print("Saved: epsilon_curve.png")

    print("\nAll graphs saved successfully!\n")

# -----------------------------------------------
# Run from command line
# -----------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <csv_file>")
        sys.exit(0)

    plot_csv(sys.argv[1])