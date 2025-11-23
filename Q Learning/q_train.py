# multi_alpha_train.py : Train Q-Learning 2048 for multiple alpha (learning rates)
import random, pickle, os
import matplotlib.pyplot as plt
from collections import deque
from env import Game2048   # uses your same environment

# ==================== GLOBAL SETTINGS ====================
alphas      = [0.05, 0.10, 0.15, 0.25]   # << try different learning rates here!
gamma       = 0.98
epsilon     = 1.0
epsilon_min = 0.10
epsilon_decay = 0.9993
episodes    = 20000

# ==================== CANONICAL REPRESENTATION ====================
def canonical(board):
    b = tuple(board)
    r = (
        tuple(reversed(board[:4])) +
        tuple(reversed(board[4:8])) +
        tuple(reversed(board[8:12])) +
        tuple(reversed(board[12:16]))
    )
    return min(b, r)

# ==================== TRAIN ONE α ====================
def train_for_alpha(alpha):
    """Trains a Q-Learning agent for a given alpha and returns its learning curve."""
    Q = {}
    scores_window = deque(maxlen=1000)
    curve = []
    eps = epsilon

    def get_Q(s, a): return Q.get((s, a), 0.0)

    for ep in range(1, episodes + 1):
        game = Game2048()

        while True:
            s = canonical(game.board)
            legal = game.legal_moves()
            if not legal: break

            # ε-greedy policy
            if random.random() < eps:
                a = random.choice(legal)
            else:
                vals = [get_Q(s, x) for x in legal]
                a = legal[vals.index(max(vals))]

            reward = game.move(a)

            # Reward shaping
            if reward < 0: reward = -10
            reward += 2 * game.score
            reward += 50 * sum(1 for x in game.board if x == 0)

            game.popup()
            done = len(game.legal_moves()) == 0
            s2 = canonical(game.board)

            # Q-Learning update
            old = get_Q(s, a)
            if not done:
                future = max(get_Q(s2, na) for na in game.legal_moves())
            else:
                future = -200
            Q[(s, a)] = old + alpha * (reward + gamma * future - old)

            if done: break

        scores_window.append(game.score)

        # Track curve every 50 episodes
        if ep % 50 == 0:
            curve.append(sum(scores_window) / len(scores_window))

        # Logging average of last 500
        if ep % 500 == 0:
            last500 = list(scores_window)[-500:] if len(scores_window) >= 500 else list(scores_window)
            avg500 = sum(last500) / len(last500)
            print(f"[α={alpha}] Episode {ep} | ε={eps:.3f} | Avg (last 500)={avg500:.2f}")

        eps = max(eps * epsilon_decay, epsilon_min)

    # Save Q-table for this α
    filename = f"qtable_alpha_{alpha}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(Q, f)
    print(f"Saved Q-table -> {filename}")

    return curve

# ==================== MAIN MULTI-ALPHA LOOP ====================
if __name__ == "__main__":
    plt.figure(figsize=(10,6))

    for a in alphas:
        print(f"\n=== Training for alpha = {a} ===")
        curve = train_for_alpha(a)
        plt.plot(range(50, 50*len(curve)+1, 50), curve, label=f"α = {a}")

    plt.title("Q-Learning 2048: Effect of Learning Rate α")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Score (Last 1000)")
    plt.legend()
    plt.grid(True)
    plt.savefig("multi_alpha_q_curve.png")
    print("\nSaved plot as multi_alpha_q_curve.png")
    plt.show()