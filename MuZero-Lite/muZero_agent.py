# muzero_agent.py
import torch
import numpy as np

from env import Game2048
from muzero_lite import RepresentationNet, DynamicsNet, PredictionNet
from muzero_lite import DEVICE, ACTION_SIZE, board_to_tensor


class MuZeroAgent:
    """Inference-time MuZero agent for 2048 using trained networks."""
    def __init__(self, checkpoint_path, device=None):
        self.device = device or DEVICE

        # Build SAME networks as used in training
        self.repr = RepresentationNet().to(self.device)
        self.dyn  = DynamicsNet().to(self.device)
        self.pred = PredictionNet().to(self.device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        ms = ckpt["model_state"]

        # Load correct weights
        self.repr.load_state_dict(ms["repr"])
        self.dyn.load_state_dict(ms["dyn"])
        self.pred.load_state_dict(ms["pred"])

        self.repr.eval()
        self.dyn.eval()
        self.pred.eval()

        print(f"[MuZeroAgent] Loaded checkpoint: {checkpoint_path}")

    # -------------------------------------------------------------
    # Inference: Use representation + prediction only (no MCTS)
    # -------------------------------------------------------------
    def act(self, game):
        legal = game.legal_moves()
        if not legal:
            return None

        x = board_to_tensor(game.board).unsqueeze(0)  # (1,16)

        with torch.no_grad():
            latent = self.repr(x)
            logits, value = self.pred(latent)
            logits = logits.squeeze(0).cpu().numpy()

        # Mask illegal moves
        masked = np.full(ACTION_SIZE, -1e9, dtype=np.float32)
        for a in legal:
            masked[a] = logits[a]

        return int(np.argmax(masked))