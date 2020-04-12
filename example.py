#!/usr/bin/env python
# encoding: utf-8

import gym
import pprint
import ray
import ray.rllib.agents.ppo as ppo

CHECKPOINT_PATH = "/tmp/ppo/cart"
SELECT_ENV = "CartPole-v0"
N_ITER = 10


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


def rollout_actions (agent, env, debug=True, render=True):
    state = env.reset()
    done = False
    cumulative_reward = 0

    while not done:
        last_state = state
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        cumulative_reward += reward

        if debug:
            print("state", last_state, "action", action, "reward", reward)

        if render:
            env.render()

    return cumulative_reward


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    # train a policy with RLlib using PPO

    agent = ppo.PPOTrainer(config, env=SELECT_ENV)
    checkpoint_path, reward_history = train_policy(agent, CHECKPOINT_PATH)

    print(reward_history)

    # apply the trained policy in a use case

    agent.restore(checkpoint_path)
    env = gym.make(SELECT_ENV)
    cumulative_reward = rollout_actions(agent, env)

    print(cumulative_reward)
