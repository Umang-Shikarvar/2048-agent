# multi_alpha_dql_train.py : Double Q-Learning benchmark for multiple alpha values
import random, pickle, os
import matplotlib.pyplot as plt
from collections import deque
from env import Game2048

# ==================== GLOBAL SETTINGS ====================
alphas       = [0.05, 0.10, 0.15, 0.25]  # << change alpha values here
gamma        = 0.98
epsilon      = 1.0
epsilon_min  = 0.10
epsilon_decay = 0.99935
episodes     = 20000

# ==================== STATE COMPRESSION ====================
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
    """Train Double Q-learning for a given alpha and return learning curve."""
    Q1, Q2 = {}, {}
    scores_window = deque(maxlen=1000)  # for averaging
    curve = []
    eps = epsilon

    def Q(qt, s, a): return qt.get((s, a), 0.0)
    def Q_total(s, a): return Q1.get((s, a), 0.0) + Q2.get((s, a), 0.0)

    for ep in range(1, episodes + 1):
        game = Game2048()

        while True:
            s = canonical(game.board)
            legal = game.legal_moves()
            if not legal: break

            # epsilon-greedy using Q1+Q2 combined
            if random.random() < eps:
                a = random.choice(legal)
            else:
                vals = [Q_total(s, x) for x in legal]
                a = legal[vals.index(max(vals))]

            # -------- before state info --------
            empty_before = sum(1 for x in game.board if x == 0)
            merge_reward = game.move(a)

            # reward shaping
            if merge_reward < 0:
                reward = -20
            else:
                empty_after = sum(1 for x in game.board if x == 0)
                delta_empty = empty_after - empty_before
                reward = merge_reward + 5 * delta_empty - 1
                game.popup()

            done = len(game.legal_moves()) == 0
            s2 = canonical(game.board)

            # -------- Double Q update --------
            if random.random() < 0.5:
                old = Q(Q1, s, a)
                if not done:
                    a2 = max(game.legal_moves(), key=lambda na: Q(Q1, s2, na))
                    future = Q(Q2, s2, a2)
                else:
                    future = 0
                Q1[(s, a)] = old + alpha * (reward + gamma * future - old)
            else:
                old = Q(Q2, s, a)
                if not done:
                    a1 = max(game.legal_moves(), key=lambda na: Q(Q2, s2, na))
                    future = Q(Q1, s2, a1)
                else:
                    future = 0
                Q2[(s, a)] = old + alpha * (reward + gamma * future - old)

            if done:
                break

        scores_window.append(game.score)

        # ---- Track curve every 50 episodes ----
        if ep % 50 == 0:
            curve.append(sum(scores_window)/len(scores_window))

        # ---- Logging every 500 episodes ----
        if ep % 500 == 0:
            last500 = (list(scores_window)[-500:]
                       if len(scores_window) >= 500 else list(scores_window))
            print(f"[DQL α={alpha}] Episode {ep} | ε={eps:.3f} | Avg(last 500)={sum(last500)/len(last500):.2f}")

        eps = max(eps * epsilon_decay, epsilon_min)

    # ---- Save Q1+Q2 ----
    filename = f"dql_qtable_alpha_{alpha}.pkl"
    with open(filename, "wb") as f:
        pickle.dump((Q1, Q2), f)
    print(f"Saved Double Q-table -> {filename}")

    return curve

# ==================== MAIN LOOP ====================
if __name__ == "__main__":
    plt.figure(figsize=(10,6))

    for a in alphas:
        print(f"\n=== Training Double Q-Learning for α = {a} ===")
        curve = train_for_alpha(a)
        plt.plot(range(50, 50*len(curve)+1, 50), curve, label=f"α = {a}")

    plt.title("Double Q-Learning 2048: Effect of Learning Rate α")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Score (Last 1000)")
    plt.legend()
    plt.grid(True)
    plt.savefig("multi_alpha_dql_curve.png")
    print("\nSaved plot -> multi_alpha_dql_curve.png")
    plt.show()