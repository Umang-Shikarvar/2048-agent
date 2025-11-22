# train_sarsa_q.py
import random
import logging
import time
import matplotlib.pyplot as plt
from collections import deque

from env import Game2048
from sarsa_qagent import SarsaQNTupleAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger("train")

def evaluate(agent, episodes=200):
    total = 0
    mx = 0

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
        mx = max(mx, g.score)

    return total / episodes, mx

def train_sarsa_q(
    episodes=20000,
    alpha_list=(0.001, 0.005, 0.01),
    gamma=0.99,
    epsilon=0.05,
    eval_every=200,
    eval_episodes=100
):

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Classical SARSA-Q N-Tuple Training Progress")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Smoothed Eval Average Score")

    curves = {}
    windows = {}
    lines = {}

    for alpha in alpha_list:
        curves[alpha] = ([], [])
        windows[alpha] = deque(maxlen=20)
        line, = ax.plot([], [], label=f"alpha={alpha}")
        lines[alpha] = line

    ax.legend()


    for alpha in alpha_list:
        logger.info(f"\n============================")
        logger.info(f" TRAINING SARSA-Q alpha={alpha}")
        logger.info(f"============================")

        agent = SarsaQNTupleAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

        t0 = time.time()

        for ep in range(1, episodes + 1):
            game = Game2048()
            while True:
                legal = game.legal_moves()
                if not legal:
                    break

                a = agent.act(game)
                r = game.move(a)
                if r < 0:
                    break

                game.popup()
                s_after = list(game.board)

                next_legal = game.legal_moves()
                if not next_legal:
                    agent.update(s_after, a, r, None, None, terminal=True)
                    break

                # pick next action a'
                a_next = agent.act(game)

                # simulate next action to get next-afterstate
                temp = Game2048()
                temp.board = list(game.board)
                r2 = temp.move(a_next)
                s_next_after = temp.board if r2 >= 0 else None

                # SARSA-Q update
                agent.update(
                    s_after,
                    a,
                    r,
                    s_next_after,
                    a_next,
                    terminal=(s_next_after is None)
                )

            if ep % eval_every == 0:
                avg, mx = evaluate(agent, episodes=eval_episodes)
                windows[alpha].append(avg)
                smooth_avg = sum(windows[alpha]) / len(windows[alpha])

                x_data, y_data = curves[alpha]
                x_data.append(ep)
                y_data.append(smooth_avg)

                lines[alpha].set_xdata(x_data)
                lines[alpha].set_ydata(y_data)

                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

                elapsed = time.time() - t0
                logger.info(f"alpha={alpha} ep={ep}/{episodes} avg={avg:.1f} smooth={smooth_avg:.1f} elapsed={elapsed:.1f}s")

        # Save Q-table for each alpha
        agent.save(f"sarsa_q_alpha_{alpha}.pkl")
        logger.info(f"Saved: sarsa_q_alpha_{alpha}.pkl")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train_sarsa_q(
        episodes=20000,
        alpha_list=(0.001, 0.005, 0.01),
        eval_every=200,
        eval_episodes=100
    )
