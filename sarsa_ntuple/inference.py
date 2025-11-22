"""
Benchmark multiple agents on 2048 environment (NO UI)
Runs 1000 episodes for each agent and prints formatted metrics.
"""

import random
import logging
from env import Game2048
from random_agent import RandomAgent
from sarsa_agent import SarsaNTupleAgent
# from q_agent import QAgent  

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("inference")

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
            if action not in legal:
                action = random.choice(legal)

            reward = game.move(action)
            if reward < 0:
                break

            game.popup()

        scores.append(game.score)
        max_tiles.append(max(game.board))

    avg_score = sum(scores) / episodes
    max_score = max(scores)

    return avg_score, max_score, max_tiles


def print_results(name, avg_score, max_score, tiles):
    print("\n========================================")
    print(f"AGENT: {name}")
    print("========================================")

    print(f"Average Score : {avg_score:.2f}")
    print(f"Max Score     : {max_score}")

    # --- Distribution ---
    freq = {}
    for t in tiles:
        v = (1 << t) if t > 0 else 0
        freq[v] = freq.get(v, 0) + 1

    print("\n Max Tile Distribution:")
    for v, c in sorted(freq.items()):
        print(f"  Tile {v:<4}: {c:>4} times ({100*c/len(tiles):.2f}%)")

    # --- Win Rate ---
    wins = sum(c for v, c in freq.items() if v >= 2048)
    print(f"\n Win Rate (>= 2048): {100*wins/len(tiles):.2f}%")

    print("========================================\n")


if __name__ == "__main__":
    agents = {
        "RandomAgent": RandomAgent(),
        "SARSA N-tuple Agent": SarsaNTupleAgent(alpha=0.01, epsilon=0.01),
        # "Q-Learning Agent": QAgent(),
    }

    for name, agent in agents.items():
        avg, mx, tiles = evaluate_agent(agent, episodes=500)
        print_results(name, avg, mx, tiles)
