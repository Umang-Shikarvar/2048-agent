# q_train.py (IMPROVED Q-Learning Baseline for 2048 with Plots)
import random, pickle, os
from env import Game2048
import matplotlib.pyplot as plt
from collections import deque

# ==================== HYPERPARAMETERS ====================
alpha        = 0.15
gamma        = 0.98
epsilon      = 1.0
epsilon_min  = 0.10
epsilon_decay= 0.9993
episodes     = 25000

# ==================== Q-TABLE STORAGE ====================
Q = {}  # (state, action) -> Q-value

# ==================== TRACKING METRICS ====================
scores_window = deque(maxlen=1000)
avg_scores = []          # average curve
eps_track  = []          # epsilon curve
episodes_x = []          # x-axis steps

# ==================== STATE REPRESENTATION ====================
def canonical(board):
    """Reduce state space using symmetry (mirror)."""
    b = tuple(board)
    r = (
        tuple(reversed(board[:4])) + \
        tuple(reversed(board[4:8])) + \
        tuple(reversed(board[8:12])) + \
        tuple(reversed(board[12:16]))
    )
    return min(b, r)

def get_state(board): return canonical(board)
def get_Q(s, a): return Q.get((s, a), 0.0)

# ==================== ACTION SELECTION ====================
def best_action(game: Game2048):
    s = get_state(game.board)
    legal = game.legal_moves()
    values = [get_Q(s, a) for a in legal]
    return legal[values.index(max(values))]

# ==================== TRAINING LOOP ====================
def train():
    global epsilon, Q

    # ----- Optional: Resume Training -----
    if os.path.exists("ql_qtable.pkl"):
        with open("ql_qtable.pkl", "rb") as f:
            Q = pickle.load(f)
        print("Loaded checkpoint ql_qtable.pkl")

    for ep in range(1, episodes + 1):
        game = Game2048()

        while True:
            s = get_state(game.board)
            legal = game.legal_moves()
            if not legal: break

            # ε-greedy policy
            if random.random() < epsilon:
                a = random.choice(legal)
            else:
                a = best_action(game)

            # ----- Take action -----
            reward = game.move(a)

            # ----- Reward shaping -----
            if reward < 0: reward = -10                      # invalid moves punished
            reward += 2 * game.score                         # reward scoring
            reward += 50 * sum(1 for x in game.board if x==0)  # reward empty tiles

            game.popup()
            done = (len(game.legal_moves()) == 0)
            s2 = get_state(game.board)

            # ----- Q-Learning update -----
            old = get_Q(s, a)
            if not done:
                future = max(get_Q(s2, na) for na in game.legal_moves())
            else:
                future = -200      # penalty for losing early

            Q[(s, a)] = old + alpha * (reward + gamma * future - old)

            if done: break

        # ====== Tracking ======
        scores_window.append(game.score)

        if ep % 50 == 0:     # update curve less frequently
            avg_scores.append(sum(scores_window) / len(scores_window))
            eps_track.append(epsilon)
            episodes_x.append(ep)
            live_plot()

        # ----- Update epsilon -----
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # ----- Logging (avg of last 500) -----
        if ep % 500 == 0:
            last500 = list(scores_window)[-500:] if len(scores_window) >= 500 else list(scores_window)
            avg500 = sum(last500) / len(last500)
            print(f"[Episode {ep}] epsilon={epsilon:.3f}, avg score (last 500) = {avg500:.2f}")

        # ----- Checkpoint every 5000 -----
        if ep % 5000 == 0:
            save_q()

    save_q()
    save_final_plot()
    print("Training finished! Saved -> ql_qtable.pkl")


# ==================== LIVE PLOTTING ====================
def live_plot():
    plt.clf()
    plt.title("Q-Learning 2048 Training Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Score (Last 1000)")
    plt.plot(episodes_x, avg_scores, label="Avg Score (1000-window)")
    plt.legend()
    plt.pause(0.001)


def save_final_plot():
    plt.savefig("ql_training_curve.png")
    print("Saved graph -> training_curve.png")


# ==================== SAVE FUNCTION ====================
def save_q():
    with open("ql_qtable.pkl", "wb") as f:
        pickle.dump(Q, f)
    print("Checkpoint saved!")


# ==================== MAIN ====================
if __name__ == "__main__":
    plt.ion()     # interactive plot
    train()
    plt.ioff()
    plt.show()