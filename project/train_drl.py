"""
Retrain the three FinRL DRL agents (PPO, A2C, DDPG) on a fresh train window.

Reads merged market+sentiment data from dashboard_data/merged_data.csv (built
by pipeline.py step 4) and writes new model zips into
sentiment-drl-trading-main/trained_models/ where pipeline.py loads them from.

Usage:
    python train_drl.py                          # uses TRAIN_START / TRAIN_END from pipeline.py
    python train_drl.py --train-end 2025-01-01   # override TRAIN_END only
    python train_drl.py --train-start 2022-03-02 --train-end 2025-01-01
    python train_drl.py --agents ppo,a2c         # subset of agents
    python train_drl.py --ppo-steps 75000 --a2c-steps 50000 --ddpg-steps 20000
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

# Import constants + helpers from pipeline so env config stays in lock-step
import pipeline as pl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_PATH = os.path.join(BASE_DIR, "dashboard_data", "merged_data.csv")
MODEL_DIR = os.path.join(pl.DRL_DIR, "trained_models")

SEED = 42
DEFAULT_STEPS = {"ppo": 75_000, "a2c": 50_000, "ddpg": 20_000}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0003,
    "batch_size": 128,
}
A2C_PARAMS = {
    "n_steps": 10,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
}
DDPG_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100_000,
    "learning_rate": 0.001,
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_train_env(train_df: pd.DataFrame):
    env_kwargs = pl.build_env_kwargs(train_df)
    e_train = StockTradingEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train.get_sb_env()
    env_train.seed(SEED)
    env_train.action_space.seed(SEED)
    return env_train, env_kwargs


def train_one(algo: str, env_train, total_timesteps: int, out_path: str) -> None:
    agent = DRLAgent(env=env_train)
    if algo == "ppo":
        model = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    elif algo == "a2c":
        model = agent.get_model("a2c", model_kwargs=A2C_PARAMS)
    elif algo == "ddpg":
        model = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
    else:
        raise ValueError(f"unknown algo {algo}")

    # logger (FinRL expects a results/ dir per algo)
    log_dir = os.path.join(config.RESULTS_DIR, algo)
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "csv"]))

    print(f"  training {algo.upper()} for {total_timesteps:,} steps...")
    trained = agent.train_model(
        model=model, tb_log_name=algo, total_timesteps=total_timesteps
    )
    trained.save(out_path)
    print(f"  saved {algo.upper()} → {out_path}.zip")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default=pl.TRAIN_START)
    ap.add_argument("--train-end", default=pl.TRAIN_END)
    ap.add_argument("--agents", default="ppo,a2c,ddpg",
                    help="comma-separated subset of ppo,a2c,ddpg")
    ap.add_argument("--ppo-steps", type=int, default=DEFAULT_STEPS["ppo"])
    ap.add_argument("--a2c-steps", type=int, default=DEFAULT_STEPS["a2c"])
    ap.add_argument("--ddpg-steps", type=int, default=DEFAULT_STEPS["ddpg"])
    args = ap.parse_args()

    if not os.path.exists(MERGED_PATH):
        print(f"ERROR: {MERGED_PATH} not found. Run `python pipeline.py` first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading merged data: {MERGED_PATH}")
    df = pd.read_csv(MERGED_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    print(f"Train window: {args.train_start} → {args.train_end}")
    train = data_split(df, args.train_start, args.train_end)
    print(f"  train rows: {len(train):,}  tickers: {train['tic'].nunique()}")

    set_global_seed(SEED)
    env_train, _ = build_train_env(train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    steps = {"ppo": args.ppo_steps, "a2c": args.a2c_steps, "ddpg": args.ddpg_steps}
    wanted = [a.strip().lower() for a in args.agents.split(",") if a.strip()]

    for algo in wanted:
        if algo not in steps:
            print(f"skipping unknown agent {algo!r}")
            continue
        out_path = os.path.join(MODEL_DIR, f"agent_{algo}")
        train_one(algo, env_train, steps[algo], out_path)

    print("DONE.")


if __name__ == "__main__":
    main()
