#!/usr/bin/env python
# encoding: utf-8

from gym_projectile.envs.projectile_env import Projectile_v0
from ray.tune.registry import register_env
import gym
import pprint
import ray
import ray.rllib.agents.ppo as ppo
import sys


CHECKPOINT_PATH = "/tmp/ppo/proj"
SELECT_ENV = "projectile-v0"
N_ITER = 2


def train_policy (agent, path, debug=True, n_iter=N_ITER):
    reward_history = []

    for _ in range(n_iter):
        result = agent.train()

        max_reward = result["episode_reward_max"]
        reward_history.append(max_reward)

        checkpoint_path = agent.save(path)

        if debug:
            pprint.pprint(result)

    return checkpoint_path, reward_history


def rollout_actions (agent, env, debug=True, render=True, max_steps=100):
    state = env.reset()

    for step in range(max_steps):
        last_state = state
        action = agent.compute_action(state, explore=True)
        state, reward, done, _ = env.step(action)

        if debug:
            print("state", last_state, "action", action, "reward", reward)

        if render:
            env.render()

        if done == 1 and reward > 0:
            break


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    register_env("projectile-v0", lambda config: Projectile_v0())
    env = gym.make(SELECT_ENV)

    # train a policy with RLlib using PPO

    agent = ppo.PPOTrainer(config, env=SELECT_ENV)
    checkpoint_path, reward_history = train_policy(agent, CHECKPOINT_PATH)

    print(reward_history)

    # apply the trained policy in a use case

    agent.restore(checkpoint_path)
    rollout_actions(agent, env)
