import random

class RandomAgent:
    """Default agent if nothing provided: picks any legal move randomly"""
    def act(self, game):
        return random.choice(game.legal_moves())