#!/usr/bin/env python
# encoding: utf-8

import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil

## start Ray

ray.shutdown()
ray.init(ignore_reinit_error=True)

## set up directories to log results and checkpoints

CHECKPOINT_ROOT = "tmp/ppo/cart"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

## configure the CartPole environment

SELECT_ENV = "CartPole-v1"

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

agent = ppo.PPOTrainer(config, env=SELECT_ENV)

## train a policy with PPO

N_ITER = 30
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

for n in range(N_ITER):
    result = agent.train()
    file_name = agent.save(CHECKPOINT_ROOT)

    print(s.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        file_name
        ))

## examine the trained policy

policy = agent.get_policy()
model = policy.model
print(model.base_model.summary())
