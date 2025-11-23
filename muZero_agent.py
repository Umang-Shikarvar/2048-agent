# muzero_agent.py
import torch
import numpy as np
from env import Game2048

# Import EXACT SAME architectures from training
from muzero_lite import RepresentationNet, DynamicsNet, PredictionNet
from muzero_lite import DEVICE, ACTION_SIZE, board_to_tensor, one_hot_action


class MuZeroAgent:
    """Inference-time MuZero agent for 2048 using trained networks."""
    def __init__(self, checkpoint_path, device=None):
        self.device = device or DEVICE

        # Rebuild SAME networks as training
        self.repr = RepresentationNet().to(self.device)
        self.dyn  = DynamicsNet().to(self.device)
        self.pred = PredictionNet().to(self.device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_state = ckpt["model_state"]

        self.repr.load_state_dict(model_state["repr"])
        self.dyn.load_state_dict(model_state["dyn"])
        self.pred.load_state_dict(model_state["pred"])

        self.repr.eval()
        self.dyn.eval()
        self.pred.eval()

        print(f"[MuZeroAgent] Loaded checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------
    # MuZero inference: 1-step lookahead (fast & simple)
    # ------------------------------------------------------------
    def act(self, game):
        legal = game.legal_moves()
        if not legal:
            return None

        # Encode board
        x = board_to_tensor(game.board).unsqueeze(0)  # (1,16)

        with torch.no_grad():
            latent = self.repr(x)                      # (1,latent)
            logits, value = self.pred(latent)
            logits = logits.squeeze(0).cpu().numpy()

        # Mask illegal actions
        masked = np.full(ACTION_SIZE, -1e9, dtype=np.float32)
        for a in legal:
            masked[a] = logits[a]

        action = int(np.argmax(masked))
        return action