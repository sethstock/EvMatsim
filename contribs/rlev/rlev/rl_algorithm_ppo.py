"""
This script implements a reinforcement learning (RL) training pipeline using
the Proximal Policy Optimization (PPO) algorithm from the Stable-Baselines3
library. The training is performed on custom Matsim-based environments, which
can use an MLP, a GNN, or a GraphGPS encoder as the policy backbone.

It supports parallelized environments, custom callbacks for TensorBoard logging
and checkpointing, and configurable hyperparameters. It also allows resuming
training from a previously saved model.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Optional imports only used for type hints/logging of best env
from rlev.envs.matsim_graph_env_gnn import MatsimGraphEnvGNN  # noqa: F401
from rlev.envs.matsim_graph_env_mlp import MatsimGraphEnvMlp  # noqa: F401

# GraphGPS extractor (ensure this file exists: rlev/models/graphgps_extractor.py)
from .graphgps_extractor import GraphGPSExtractor

def _print_cuda_diag():
    try:
        import torch
        info = dict(
            cuda_available=torch.cuda.is_available(),
            torch_cuda=torch.version.cuda,
            n_devices=torch.cuda.device_count(),
            current_device=(torch.cuda.current_device() if torch.cuda.is_available() else None),
            device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        )
        print(f"[CUDA DIAG] {info}", flush=True)
    except Exception as e:
        print(f"[CUDA DIAG] error: {e}", flush=True)


class TensorboardCallback(BaseCallback):
    """
    Logs average metrics and keeps track of the best-performing env snapshot.
    Expects each env to put {"graph_env_inst": <env_instance>} into info dicts.
    """

    def __init__(self, verbose: int = 0, save_dir: str | None = None):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.best_reward = -np.inf
        # Keep the type annotation loose to support all env wrappers (MLP/GNN/GPS)
        self.best_env: Any = None

    def _on_step(self) -> bool:
        avg_reward = 0.0
        avg_cost = 0.0
        avg_charger_eff = 0.0
        avg_time_eff = 0.0

        infos_list = self.locals.get("infos", [])
        n = len(infos_list) if infos_list else 0

        for i, infos in enumerate(infos_list):
            env_inst = infos.get("graph_env_inst", None)
            if env_inst is None:
                continue

            reward = float(getattr(env_inst, "_reward", 0.0))
            avg_reward += reward
            avg_cost += float(getattr(env_inst, "_charger_cost", 0.0))
            avg_charger_eff += float(getattr(env_inst, "_charger_efficiency", 0.0))
            avg_time_eff += float(getattr(env_inst, "_time_efficiency", 0.0))

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_env = env_inst
                # Persist best snapshot
                if self.save_dir is not None:
                    self.best_env.save_charger_config_to_csv(Path(self.save_dir, "best_chargers.csv"))
                    if getattr(self.best_env, "best_output_response", None) is not None:
                        self.best_env.save_server_output(self.best_env.best_output_response, "bestoutput")

        if n > 0:
            self.logger.record("metrics/avg_reward", avg_reward / n)
            self.logger.record("metrics/best_reward", self.best_reward)
            self.logger.record("metrics/avg_charger_cost", avg_cost / n)
            self.logger.record("metrics/avg_charger_efficiency", avg_charger_eff / n)
            self.logger.record("metrics/avg_time_efficiency", avg_time_eff / n)

        return True


def main(args: argparse.Namespace):
    # Output directory for this run
    save_dir = f"{args.results_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}/"
    os.makedirs(save_dir, exist_ok=True)

    # Persist CLI args for reproducibility
    with open(Path(save_dir, "args.txt"), "w") as f:
        for key, val in vars(args).items():
            f.write(f"{key}:{val}\n")

    # --------- Env factory ---------
    def make_env():
        if args.policy_type == "MlpPolicy":
            return gym.make(
                "MatsimGraphEnvMlp-v0",
                config_path=args.matsim_config,
                num_agents=args.num_agents,
                save_dir=save_dir,
            )
        elif args.policy_type == "GNNPolicy":
            return gym.make(
                "MatsimGraphEnvGNN-v0",
                config_path=args.matsim_config,
                num_agents=args.num_agents,
                save_dir=save_dir,
            )
        elif args.policy_type == "GraphGPS":
            # Requires your GraphGPS env wrapper to be registered as MatsimGraphEnvGPS-v0
            return gym.make(
                "MatsimGraphEnvGPS-v0",
                config_path=args.matsim_config,
                num_agents=args.num_agents,
                save_dir=save_dir,
                pe_dim=0,  # set >0 once you add positional encodings
            )
        else:
            raise ValueError(f"Unknown policy_type: {args.policy_type}")

    # Windows-friendly vectorized env (threads, not processes)
    env = DummyVecEnv([make_env for _ in range(args.num_envs)])

    # The save frequency only accounts for how many times each env has run,
    # so divide to save every args.save_frequency *total* timesteps.
    args.save_frequency //= max(1, args.num_envs)

    tensorboard_cb = TensorboardCallback(save_dir=save_dir)
    checkpoint_cb = CheckpointCallback(save_freq=args.save_frequency, save_path=save_dir)
    callback = CallbackList([tensorboard_cb, checkpoint_cb])

    # --------- Policy selection + kwargs ---------
    _print_cuda_diag()

    if args.device == "cuda":
        device_str = "cuda:0"  # will raise if CUDA is actually unavailable
    elif args.device == "cpu":
        device_str = "cpu"
    else:
    # auto
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[RL] Using device={device_str}", flush=True)


    # Default: MLP / legacy GNN path
    policy_id = args.policy_type
    policy_kwargs: dict[str, Any] = dict(net_arch=args.mlp_dims)

    # GraphGPS path uses a custom features extractor and MultiInputPolicy
    if args.policy_type == "GraphGPS":
        policy_id = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=GraphGPSExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                num_layers=3,
                heads=4,
                dropout=0.2,
                attn_dropout=0.2,
                fixed_k=128,     # lower to 48/32 if still tight on VRAM
                use_amp=True,
                use_cudagraphs=True,
            ),
            net_arch=dict(pi=[128], vf=[128]),
            share_features_extractor=True,
        )



    # --------- Build or load model ---------
    if args.model_path:
        model = PPO.load(
            args.model_path,
            env=env,
            n_steps=args.num_steps,
            verbose=1,
            device=device_str,
            tensorboard_log=save_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            policy_kwargs=policy_kwargs,
        )
    else:
        model = PPO(
            policy_id,
            env,
            n_steps=args.num_steps,
            verbose=1,
            device=device_str,
            tensorboard_log=save_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            policy_kwargs=policy_kwargs,
        )

    # Optional: enable CUDA Graphs capture for GraphGPS after warmup (fixed shapes required)
    if args.policy_type == "GraphGPS" and torch.cuda.is_available():
        try:
            model.policy.features_extractor._use_cudagraphs = True  # capture on first forward
        except Exception:
            pass

    # --------- Train ---------
    # total_timesteps = n_steps * num_envs * iterations
    model.learn(total_timesteps=args.num_timesteps, callback=callback)
    model.save(Path(save_dir, "ppo_matsim"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO model on the MatsimGraphEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matsim_config", type=str, help="Path to the matsim config.xml file.")
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1_000_000,
        help="Total number of timesteps to train. num_timesteps = n_steps * num_envs * iterations.",
    )
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to run in parallel.")
    parser.add_argument(
        "--num_agents",
        type=int,
        default=-1,
        help=(
            "Number of vehicles to simulate in the matsim simulator. "
            "If num_agents < 0, existing plans.xml and vehicles.xml are used."
        ),
    )
    parser.add_argument(
        "--mlp_dims",
        default="256 128 64",
        help="Dimensions of the MLP layers as space-separated integers (e.g., '256 128 64').",
    )
    parser.add_argument("--results_dir", type=str, default=Path(Path(__file__).parent, "ppo_results"),
                        help="Directory to save TensorBoard logs and model checkpoints.")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of steps each environment takes before updating the policy/value.")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Batch size PPO uses when sampling from the rollout buffer for updates.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Optimizer learning rate.")
    parser.add_argument("--model_path", default=None,
                        help="Path to a saved model.zip to resume training.")
    parser.add_argument("--save_frequency", type=int, default=10_000,
                        help="How often to save model weights, in *total* timesteps.")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument(
        "--policy_type",
        default="MlpPolicy",
        choices=["MlpPolicy", "GNNPolicy", "GraphGPS"],
        type=str,
        help="Policy type / encoder backbone.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device override. 'auto' chooses cuda if available else cpu.")

    # Parse + normalize args
    args = parser.parse_args()
    # Convert "256 128 64" -> [256, 128, 64]
    args.mlp_dims = [int(x) for x in str(args.mlp_dims).split()]
    # Run
    main(args)
