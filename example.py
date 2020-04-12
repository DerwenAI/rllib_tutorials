#!/usr/bin/env python
# encoding: utf-8

import gym
import ray
import ray.rllib.agents.ppo as ppo

SELECT_ENV = "CartPole-v0"
N_ITER = 10


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    reward_history = []
    agent = ppo.PPOTrainer(config, env=SELECT_ENV)

    for _ in range(N_ITER):
        result = agent.train()
        #print(result)

        max_reward = result["episode_reward_max"]
        reward_history.append(max_reward)

        chkpt_path = agent.save("/tmp/ppo/cart")
        print(f"\n{chkpt_path}")


    agent.restore(chkpt_path)

    env = gym.make(SELECT_ENV)
    state = env.reset()
    done = False
    cumulative_reward = 0

    while not done:
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        cumulative_reward += reward

    print(cumulative_reward)
