"""
Common 2048 Environment + Minimal UI (Tkinter)
All learning agents must implement: agent.act(state)
They must return an action in {0:UP, 1:RIGHT, 2:DOWN, 3:LEFT}

Usage Example:
--------------
from env import Env2048, RandomAgent

env = Env2048(agent=RandomAgent())
env.run()  # visualize & play automatically

Author: Common environment for group projects
"""

import tkinter as tk
import random
import time

# =============================== CONSTANTS =====================================
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]   # 0,1,2,3
CELL_COLORS = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
}
CELL_TEXT_COLOR = {2: "#776e65", 4: "#776e65"}
UPDATE_MS = 350  # animation speed

# =============================== GAME LOGIC 2048 ===============================

class Game2048:
    """Low-level 2048 board logic, independent from learning"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 16  # tile values stored as log2 values (0 -> empty, 1->2, 2->4...)
        self.score = 0
        self.popup()
        self.popup()

    # ---- helper methods -----
    def at(self, i): return self.board[i]
    def set(self, i, v): self.board[i] = v

    def popup(self):
        empty = [i for i in range(16) if self.at(i) == 0]
        if empty:
            idx = random.choice(empty)
            self.set(idx, 1 if random.random() < 0.9 else 2)

    def move(self, action):
        """Returns reward or -1 if invalid move (no change)."""
        before = list(self.board)
        if action == 0: reward = self.move_up()
        elif action == 1: reward = self.move_right()
        elif action == 2: reward = self.move_down()
        elif action == 3: reward = self.move_left()
        else: return -1

        if self.board != before and reward >= 0:
            self.score += reward
            return reward
        return -1

    # ---- Directions implemented by rotating board ----
    def rotate(self):  # clockwise
        b = self.board
        self.board = [b[12], b[8], b[4], b[0], b[13], b[9], b[5], b[1],
                      b[14], b[10], b[6], b[2], b[15], b[11], b[7], b[3]]
    def rotate_back(self):  # counterclockwise
        b = self.board
        self.board = [b[3], b[7], b[11], b[15], b[2], b[6], b[10], b[14],
                      b[1], b[5], b[9], b[13], b[0], b[4], b[8], b[12]]

    def slide_and_merge(self, row):
        row = [v for v in row if v]
        reward = 0
        i = 0
        while i < len(row) - 1:
            if row[i] == row[i + 1]:
                row[i] += 1
                reward += (1 << row[i])
                row.pop(i + 1)
            i += 1
        return row + [0] * (4 - len(row)), reward

    # ---- Move helpers ----
    def move_left(self):
        reward = 0
        new = []
        for r in range(4):
            row = self.board[r * 4:r * 4 + 4]
            slid, rew = self.slide_and_merge(row)
            reward += rew
            new.extend(slid)
        self.board = new
        return reward

    def move_right(self):
        self.reflect()
        r = self.move_left()
        self.reflect()
        return r

    def move_up(self):
        self.transpose()
        r = self.move_left()
        self.transpose()
        return r

    def move_down(self):
        self.transpose()
        r = self.move_right()
        self.transpose()
        return r

    # ---- transformations ----
    def reflect(self):
        b = self.board
        for r in range(4):
            self.board[r * 4:r * 4 + 4] = reversed(b[r * 4:r * 4 + 4])

    def transpose(self):
        b = self.board
        self.board = [b[0], b[4], b[8], b[12], b[1], b[5], b[9], b[13],
                      b[2], b[6], b[10], b[14], b[3], b[7], b[11], b[15]]

    def legal_moves(self):
        """returns list of valid actions"""
        valid = []
        for a in range(4):
            copy = Game2048()
            copy.board = list(self.board)
            if copy.move(a) >= 0:
                valid.append(a)
        return valid

# =============================== UI + AGENT LOOP ===============================

class Env2048:
    """UI + loop that queries an agent for actions"""

    def __init__(self, agent):
        self.game = Game2048()
        self.agent = agent

    def draw(self):
        for i, tile in enumerate(self.tiles):
            v = (1 << self.game.at(i)) if self.game.at(i) else 0
            tile.config(text=str(v if v else ""),
                        bg=CELL_COLORS.get(v, "#3c3a32"),
                        fg=CELL_TEXT_COLOR.get(v, "white"))
            tile.grid(row=i // 4, column=i % 4, padx=5, pady=5)
        self.score_label.config(text=f"Score: {self.game.score}")

    def step(self):
        legal = self.game.legal_moves()
        if not legal:
            self.score_label.config(text=f"GAME OVER! Score: {self.game.score}")
            return
        action = self.agent.act(self.game)  # MUST RETURN 0-3
        if action not in legal:
            action = random.choice(legal)   # safe fallback
        self.game.move(action)
        self.game.popup()
        self.draw()
        self.window.after(UPDATE_MS, self.step)

    def run(self):
        self.window = tk.Tk()
        self.window.title("Unified 2048 Environment")
        frame = tk.Frame(self.window, bg="#bbada0")
        frame.grid(padx=10, pady=10)
        self.score_label = tk.Label(self.window, text="Score: 0", font=("Arial", 18))
        self.score_label.grid()
        self.tiles = [tk.Label(frame, width=4, height=2, font=("Arial", 28, "bold"))
                      for _ in range(16)]
        self.draw()
        self.window.after(UPDATE_MS, self.step)
        self.window.mainloop()

# =============================== SAMPLE AGENT ==================================

class RandomAgent:
    """Default agent if nothing provided: picks any legal move randomly"""
    def act(self, game: Game2048):
        return random.choice(game.legal_moves())

# =============================== MAIN (TEST) ===================================

if __name__ == "__main__":
    env = Env2048(agent=RandomAgent())
    env.run()