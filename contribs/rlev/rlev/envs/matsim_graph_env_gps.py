import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Dict, Any, List

import torch

from .matsim_graph_env import MatsimGraphEnv

class MatsimGraphEnvGPS(MatsimGraphEnv):
    """
    GraphGPS-friendly env: observation is a Dict with 'nodes' and 'edge_index'.
    Uses the same action semantics as MatsimGraphEnv (one charger type per link).
    """

    def __init__(self, config_path, num_agents=100, save_dir=None, pe_dim: int = 0):
        super().__init__(config_path, num_agents=num_agents, save_dir=save_dir)

        # Node features (use dataset.linegraph.x): shape (N, F)
        node_shape = tuple(self.dataset.linegraph.x.shape)      # (N, F)
        self._N, self._F = node_shape

        # Edges (fixed for the scenario). PyG expects int64 edge_index
        self._edge_index = self.dataset.linegraph.edge_index.to(torch.int64)
        eidx_np = self._edge_index.cpu().numpy()
        self._E = eidx_np.shape[1]

        # Optional positional encodings (zero-initialized here; you can precompute later)
        self._pe_dim = int(pe_dim)
        self._pe = np.zeros((self._N, self._pe_dim), dtype=np.float32) if self._pe_dim > 0 else None

        # Observation space as Dict: nodes + edge_index (+ pe optional)
        spaces_dict = {
            "nodes": spaces.Box(low=0.0, high=1.0, shape=node_shape, dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=max(eidx_np.max(), 1), shape=(2, self._E), dtype=np.int64),
        }
        if self._pe_dim > 0:
            spaces_dict["pe"] = spaces.Box(low=-np.inf, high=np.inf, shape=(self._N, self._pe_dim), dtype=np.float32)

        self.observation_space = spaces.Dict(spaces_dict)

        # Action space is inherited (MultiDiscrete over links)
        # Done flag lifecycle
        self._episode_steps = 0

    # ---- helpers -----------------------------------------------------------

    def _current_obs(self) -> Dict[str, Any]:
        obs = {
            "nodes": self.dataset.linegraph.x.cpu().numpy().astype(np.float32),
            "edge_index": self._edge_index.cpu().numpy().astype(np.int64),
        }
        if self._pe_dim > 0:
            obs["pe"] = self._pe
        return obs

    # ---- gym API -----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.reward = 0.0
        self._episode_steps = 0
        return self._current_obs(), {}

    def step(self, actions: np.ndarray):
        """
        actions: shape (num_links,) with values in [0, num_charger_types-1]
        """
        # clamp/sanitize
        actions = np.asarray(actions, dtype=np.int64).clip(0, self.num_charger_types - 1)

        # compute reward via server
        reward = self.send_reward_request(actions)

        self.reward = float(reward)
        self._episode_steps += 1

        # Single-step episodes (like your  n_steps default of 1)
        terminated = False
        truncated = False
        info = {"graph_env_inst": self}

        return self._current_obs(), self.reward, terminated, truncated, info
