#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo
import shutil
import sys
import watermark

print(watermark.watermark())


checkpoint_root = "tmp/ppo/cart"
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

ray_results = f'{os.getenv("HOME")}/ray_results/'
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

info = ray.init(ignore_reinit_error=True)


SELECT_ENV = "CartPole-v1"
N_ITER = 10

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"

config["num_workers"] = 1
config["num_sgd_iter"] = 10
config["sgd_minibatch_size"] = 250
config["model"]["fcnet_hiddens"] = [100, 50]
config["num_cpus_per_worker"] = 0


agent = ppo.PPOTrainer(config, env=SELECT_ENV)

results = []
episode_data = []
episode_json = []

for n in range(N_ITER):
    result = agent.train()
    results.append(result)
    
    episode = {
        "n": n,
        "episode_reward_min": result["episode_reward_min"],
        "episode_reward_mean": result["episode_reward_mean"], 
        "episode_reward_max": result["episode_reward_max"],  
        "episode_len_mean": result["episode_len_mean"],
    }
    
    episode_data.append(episode)
    episode_json.append(json.dumps(episode))
    file_name = agent.save(checkpoint_root)
    
    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}')


ray.shutdown()
