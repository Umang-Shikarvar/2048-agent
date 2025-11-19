# q_train.py (IMPROVED Q-Learning Baseline for 2048)
import random, pickle, os
from env import Game2048

# ==================== HYPERPARAMETERS ====================
alpha        = 0.15      # Faster learning
gamma        = 0.98      # Less optimism
epsilon      = 1.0       # Exploration start
epsilon_min  = 0.10      # Still explore even late
epsilon_decay= 0.9993    # Slow decay
episodes     = 50000     # More training improves scores

# ==================== Q-TABLE STORAGE ====================
Q = {}  # (state, action) -> Q-value


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
        print("Loaded checkpoint ql_table.pkl")

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
            if reward < 0: reward = -10                     # harsh penalty for invalids
            reward += 2 * game.score                        # encourage scoring merges
            reward += 50 * sum(1 for x in game.board if x==0)  # encourage empty tiles

            game.popup()
            done = (len(game.legal_moves()) == 0)
            s2 = get_state(game.board)

            # ----- Q-Learning update -----
            old = get_Q(s, a)
            if not done:
                future = max(get_Q(s2, na) for na in game.legal_moves())
            else:
                future = -200  # BIG penalty for losing early
            Q[(s, a)] = old + alpha * (reward + gamma * future - old)

            if done: break

        # ----- Update epsilon -----
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # ----- Logging -----
        if ep % 500 == 0:
            print(f"[Episode {ep}] epsilon={epsilon:.3f}, last score={game.score}")

        # ----- Checkpoint every 5000 -----
        if ep % 5000 == 0:
            save_q()

    save_q()
    print("Training finished! Saved -> ql_qtable.pkl")


# ==================== SAVE FUNCTION ====================
def save_q():
    with open("ql_qtable.pkl", "wb") as f:
        pickle.dump(Q, f)
    print("Checkpoint saved!")


# ==================== MAIN ====================
if __name__ == "__main__":
    train()