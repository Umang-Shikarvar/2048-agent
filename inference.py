"""
Benchmark multiple agents on 2048 environment (NO UI).
Runs 1000 episodes for each agent and prints performance metrics.
"""

import random
from env import Game2048
from q_agent import QAgent
from dq_agent import DQAgent
from random_agent import RandomAgent

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
# Helper: Pretty-print results
# ==========================================================

def print_results(name, avg_score, max_score, tiles):
    # tile frequency
    tile_freq = {}
    for t in tiles:
        v = (1 << t) if t > 0 else 0
        tile_freq[v] = tile_freq.get(v, 0) + 1

    print("\n========================================")
    print(f"🧠 AGENT: {name}")
    print("========================================")
    print(f"Average Score : {avg_score:.2f}")
    print(f"Max Score     : {max_score}")
    print("\n🔢 Max Tile Distribution:")
    for v, c in sorted(tile_freq.items()):
        print(f"  Tile {v:<4}: {c:>5} times ({c*100.0/len(tiles):.2f}%)")

    # Win rate 2048+
    wins = sum(c for v, c in tile_freq.items() if v >= 2048)
    print(f"\n🏆 Win Rate (>= 2048): {wins*100.0/len(tiles):.2f}%")
    print("========================================\n")


# ==========================================================
# Run benchmark on multiple agents
# ==========================================================

if __name__ == "__main__":
    agents = {
        "RandomAgent": RandomAgent(),
        "Q-Learning Agent": QAgent(),
        "Double Q-Learning Agent": DQAgent(),
    }

    for name, agent in agents.items():
        avg_score, max_score, tiles = evaluate_agent(agent, episodes=5000)
        print_results(name, avg_score, max_score, tiles)