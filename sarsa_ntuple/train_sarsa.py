import random
import logging
import time
import matplotlib.pyplot as plt
from collections import deque

from env import Game2048
from sarsa_agent import SarsaNTupleAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger("train")


def evaluate_quiet(agent, episodes=100):
    total = 0
    max_score = 0
    for _ in range(episodes):
        g = Game2048()
        while True:
            legal = g.legal_moves()
            if not legal:
                break
            a = agent.act(g)
            if a not in legal:
                a = random.choice(legal)
            r = g.move(a)
            if r < 0:
                break
            g.popup()
        total += g.score
        max_score = max(max_score, g.score)
    return total / episodes, max_score

def train(
    episodes=20000,
    alpha_list=[0.001, 0.005, 0.01],
    gamma=0.99,
    epsilon=0.05,
    eval_every=200,
    eval_episodes=100
):

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("SARSA N-tuple Training (moving eval)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Eval avg score")

    curves = {}          # {alpha: (x_data, y_data)}
    windows = {}         # for moving average
    lines = {}           
    for alpha in alpha_list:
        curves[alpha] = ([], [])
        windows[alpha] = deque(maxlen=20)  
        (line,) = ax.plot([], [], label=f"alpha={alpha}")
        lines[alpha] = line

    ax.legend()

    for alpha in alpha_list:
        logger.info(f"\n===== TRAINING SARSA alpha={alpha} =====")

        agent = SarsaNTupleAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

        for ep in range(1, episodes + 1):
            game = Game2048()

            # ------- SARSA loop -------
            while True:
                legal = game.legal_moves()
                if not legal:
                    break

                a = agent.act(game)
                r = game.move(a)
                if r < 0:
                    break

                game.popup()
                after = list(game.board)

                next_legal = game.legal_moves()
                if not next_legal:
                    agent.learn_step(after, r, None, False)
                    break

                next_a = agent.act(game)
                temp = Game2048()
                temp.board = list(game.board)
                r2 = temp.move(next_a)
                s_next_after = temp.board if r2 >= 0 else None

                agent.learn_step(after, r, s_next_after, s_next_after is not None)

            if ep % eval_every == 0:
                avg, mx = evaluate_quiet(agent, eval_episodes)

                # smoothing using moving window
                windows[alpha].append(avg)
                smooth_avg = sum(windows[alpha]) / len(windows[alpha])

                # update curve
                x_data, y_data = curves[alpha]
                x_data.append(ep)
                y_data.append(smooth_avg)

                lines[alpha].set_xdata(x_data)
                lines[alpha].set_ydata(y_data)

                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)

                logger.info(f"alpha={alpha} ep={ep}/{episodes} eval_avg={avg:.2f} smooth={smooth_avg:.2f}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    train(
        episodes=20000,
        alpha_list=[0.001, 0.005, 0.01],
        eval_every=200,
        eval_episodes=100
    )
