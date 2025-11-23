"""
Benchmark multiple agents on 2048 environment (NO UI).
Runs 1000 episodes for each agent and prints & saves performance metrics.
"""

import random, sys
from env import Game2048
from q_agent import QAgent
from dq_agent import DQAgent
from random_agent import RandomAgent
from Expectimax.expectimax_agent import ExpectimaxAgent
from ppo_agent import PPOAgent
from muzero_agent import MuZeroAgent


# ==========================================================
# Generic evaluation (used by ALL agents)
# ==========================================================

def evaluate_agent(agent, episodes=1000):
    scores = []
    max_tiles = []

    for _ in range(episodes):
        game = Game2048()
        while True:
            legal = game.legal_moves()
            if not legal:
                break
            action = agent.act(game)
            if action not in legal:   # fallback safety
                action = random.choice(legal)
            reward = game.move(action)
            if reward >= 0:
                game.popup()
            else:
                break

        scores.append(game.score)
        max_tiles.append(max(game.board))

    avg_score = sum(scores) / episodes
    max_score = max(scores)
    return avg_score, max_score, max_tiles


# ==========================================================
# Save + Print Results
# ==========================================================

def print_and_log(f, msg):
    print(msg, end="")
    f.write(msg)


def print_results(name, avg_score, max_score, tiles, f):
    # tile frequency
    tile_freq = {}
    for t in tiles:
        v = (1 << t) if t > 0 else 0
        tile_freq[v] = tile_freq.get(v, 0) + 1

    print_and_log(f, "\n========================================\n")
    print_and_log(f, f"🧠 AGENT: {name}\n")
    print_and_log(f, "========================================\n")
    print_and_log(f, f"Average Score : {avg_score:.2f}\n")
    print_and_log(f, f"Max Score     : {max_score}\n\n")
    print_and_log(f, "🔢 Max Tile Distribution:\n")
    for v, c in sorted(tile_freq.items()):
        print_and_log(f, f"  Tile {v:<4}: {c:>5} times ({c*100.0/len(tiles):.2f}%)\n")

    # Win rate 2048+
    wins = sum(c for v, c in tile_freq.items() if v >= 2048)
    print_and_log(f, f"\n🏆 Win Rate (>= 2048): {wins*100.0/len(tiles):.2f}%\n")
    print_and_log(f, "========================================\n\n")


# ==========================================================
# Run benchmark on multiple agents
# ==========================================================

if __name__ == "__main__":
    agents = {
        # "RandomAgent": RandomAgent(),
        # "Q-Learning Agent": QAgent(),
        # "Double Q-Learning Agent": DQAgent(),
        # "Expectimax Agent": ExpectimaxAgent(),
        "PPO Agent": PPOAgent(path="/Users/zainab/2048-agent/ppo_final.pt"),
        "MuZero Agent": MuZeroAgent("/Users/zainab/2048-agent/muzero_final.pt")
    }

    with open("results.txt", "w", encoding="utf-8") as f:
        for name, agent in agents.items():
            avg_score, max_score, tiles = evaluate_agent(agent, episodes=1000)
            print_results(name, avg_score, max_score, tiles, f)

    print("\n📄 Benchmark complete! Saved to results.txt\n")