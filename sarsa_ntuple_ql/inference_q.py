# inference.py  -- improved
import random
import argparse
import os
import csv
import time
import logging
from env import Game2048
from random_agent import RandomAgent
from sarsa_qagent import SarsaQNTupleAgent  # your SARSA-Q agent class

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger("inference")

def evaluate_agent(agent, episodes=1000, csv_path=None):
    scores = []
    max_tiles = []
    start = time.time()

    writer = None
    if csv_path:
        f = open(csv_path, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "max_tile", "elapsed_s"])

    for ep in range(1, episodes + 1):
        game = Game2048()
        while True:
            legal = game.legal_moves()
            if not legal:
                break
            action = agent.act(game)
            if action not in legal:
                action = random.choice(legal)
            reward = game.move(action)
            if reward >= 0:
                game.popup()
            else:
                break

        scores.append(game.score)
        max_tiles.append(max(game.board))
        if writer:
            writer.writerow([ep, game.score, max(game.board), time.time() - start])

    if writer:
        f.close()

    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    duration = time.time() - start
    logger.info(f"Eval done: episodes={episodes} avg_score={avg_score:.2f} max_score={max_score} time={duration:.1f}s")
    return avg_score, max_score, max_tiles

def print_results(name, avg_score, max_score, tiles):
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
    wins = sum(c for v, c in tile_freq.items() if v >= 2048)
    print(f"\n🏆 Win Rate (>= 2048): {wins*100.0/len(tiles):.2f}%")
    print("========================================\n")

def load_sarsa_if_needed(path):
    agent = SarsaQNTupleAgent()
    if path:
        if not os.path.exists(path):
            logger.warning(f"Requested SARSA file not found: {path}. Running with freshly initialized agent.")
        else:
            agent.load(path)
            logger.info(f"Loaded SARSA agent from {path}")
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random","sarsa"], default="sarsa",
                        help="Which single agent to evaluate (default: sarsa)")
    parser.add_argument("--both", action="store_true", help="Evaluate both Random and SARSA agents")
    parser.add_argument("--load", type=str, default=None, help="path to .pkl/.json saved Q-table for SARSA agent")
    parser.add_argument("--episodes", type=int, default=500, help="number of episodes per agent")
    parser.add_argument("--csv", action="store_true", help="save per-episode CSV logs for each agent")
    args = parser.parse_args()

    to_run = []
    if args.both:
        to_run = ["random", "sarsa"]
    else:
        to_run = [args.agent]

    for name in to_run:
        if name == "random":
            agent = RandomAgent()
            csv_path = "random_agent_eval.csv" if args.csv else None
        else:
            agent = load_sarsa_if_needed(args.load)
            csv_path = "sarsa_agent_eval.csv" if args.csv else None

        logger.info(f"Starting evaluation for agent: {name} (episodes={args.episodes})")
        avg_score, max_score, tiles = evaluate_agent(agent, episodes=args.episodes, csv_path=csv_path)
        print_results(name, avg_score, max_score, tiles)
