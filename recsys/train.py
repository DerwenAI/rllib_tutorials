#!/usr/bin/env python
# encoding: utf-8

import pdb
import sys
import traceback

from collections import defaultdict
from gym import spaces
from pathlib import Path
from ray.tune.registry import register_env
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import csv
import gym
import numpy as np
import os
import pandas as pd
import random
import ray
import ray.rllib.agents.ppo as ppo
import shutil


######################################################################
## Gym environment

class JokeRec (gym.Env):
    DENSE_SUBMATRIX = [ 5, 7, 8, 13, 15, 16, 17, 18, 19, 20 ]
    ROW_LENGTH = 100
    MAX_STEPS = ROW_LENGTH - len(DENSE_SUBMATRIX)

    NO_RATING = "99"
    MAX_RATING = 10.0
    MAX_OBS = np.sqrt(MAX_STEPS)

    REWARD_UNRATED = -0.05	# item was not rated
    REWARD_DEPLETED = -0.1	# items depleted


    def __init__ (self, config):
        # only passing strings in config; RLlib use of JSON
        # parser was throwing exceptions due to config values
        self.dense = eval(config["dense"])
        self.centers = eval(config["centers"])
        self.clusters = eval(config["clusters"])
        self.k_clusters = len(self.clusters)

        lo = np.array([np.float64(-self.MAX_OBS)] * self.k_clusters)
        hi = np.array([np.float64(self.MAX_OBS)] * self.k_clusters)

        self.observation_space = spaces.Box(lo, hi, shape=(self.k_clusters,), dtype=np.float64)
        self.action_space = spaces.Discrete(self.k_clusters)

        # load the dataset
        self.dataset = self.load_data(config["dataset"])


    def _warm_start (self):
        """
        attempt a warm start for the rec sys, by sampling
        half of the dense submatrix of most-rated items
        """
        sample_size = round(len(self.dense) / 2.0)

        for action, items in self.clusters.items():
            for item in random.sample(self.dense, sample_size):
                if item in items:
                    state, reward, done, info = self.step(action)


    def _get_state (self):
        state = np.sqrt(self.history)
        assert state in self.observation_space, state
        return state


    def reset (self):
        self.count = 0
        self.used = []
        self.depleted = 0
        self.history = [np.float64(0.0)] * self.k_clusters

        # select a random user to simulate
        self.data_row = random.choice(self.dataset)

        # attempt a warm start
        self._warm_start()

        return self._get_state()


    def step (self, action):
        assert action in self.action_space, action
        status_str = "c[item] {}, rating {}, scaled_diff {}"

        # enumerate items that haven't been recommended previously
        items = set(self.clusters[action]).difference(set(self.used))

        if len(items) < 1:
            # items depleted
            self.depleted += 1
            item = None
            reward = self.REWARD_DEPLETED
        else:
            # select a random item from the cluster
            item = random.choice(list(items))
            rating = self.data_row[item]

            if not rating:
                # the action selected an unrated item
                reward = self.REWARD_UNRATED

            else:
                # the action selected a rated item
                reward = rating
                self.used.append(item)

                # update history: evolving distance to each cluster
                for i in range(len(self.history)):
                    c = self.centers[i]
                    scaled_diff = abs(c[item] - rating) / 2.0
                    assert scaled_diff <= 1.0, status_str.format(c[item], rating, scaled_diff)
                    self.history[i] += scaled_diff ** 2.0
                    assert np.sqrt(self.history[i]) < self.MAX_OBS, (np.sqrt(self.history[i]), scaled_diff ** 2.0)

        self.count += 1
        done = self.count >= self.MAX_STEPS
        info = { "item": item, "count": self.count, "depleted": self.depleted }

        return self._get_state(), reward, done, info


    def render (self, mode="human"):
        #print(">> ", self.data_row)
        print(">> depl:", self.depleted)

        last_used = self.used[-10:]
        last_used.reverse()
            
        print(">> used:", last_used)
        print(">> dist:", [round(np.sqrt(x), 2) for x in self.history])


    @classmethod
    def load_data (cls, data_path):
        """load the training data

        Jester collaborative filtering dataset (online joke recommender)
        https://goldberg.berkeley.edu/jester-data/

        This data file contains anonymous ratings from 24,983 users who 
        have rated 36 or more jokes.

        This is organized as a matrix with dimensions 24983 X 101

          * one row per user
          * first column gives the number of jokes rated by that user
          * the next 100 columns give the ratings for jokes 01 - 100
          * ratings are real values ranging from -10.00 to +10.00
          * the value "99" corresponds to "null" = "not rated"

        A dense sub-matrix, in which almost all users have rated the 
        jokes, includes these columns:

            {5, 7, 8, 13, 15, 16, 17, 18, 19, 20}

        See the discussion of "universal queries" in:

            Eigentaste: A Constant Time Collaborative Filtering Algorithm
            Ken Goldberg, Theresa Roeder, Dhruv Gupta, Chris Perkins
            Information Retrieval, 4(2), 133-151 (July 2001)
            https://goldberg.berkeley.edu/pubs/eigentaste.pdf
        """
        rows = []
        status_str = "input data: i {} row {} rating {}"

        with open(data_path, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")

            for row in csvreader:
                conv = [None] * (len(row) - 1)

                for i in range(1, len(row)):
                    if row[i] != cls.NO_RATING:
                        rating = float(row[i]) / cls.MAX_RATING
                        assert rating >= -1.0 and rating <= 1.0, status_str.format(i, row, rating)
                        conv[i - 1] = rating

                rows.append(conv)

        return rows


def run_one_episode (config, naive=False, verbose=False):
    env = JokeRec(config)
    env.reset()
    sum_reward = 0

    action = None
    avoid_actions = set([])
    depleted = 0

    for i in range(env.MAX_STEPS):
        if not naive or not action:
            action = env.action_space.sample()

        state, reward, done, info = env.step(action)

        if verbose:
            print("action:", action)
            print("obs:", i, state, reward, done, info)
            #env.render()

        if naive:
            if info["depleted"] > depleted:
                depleted = info["depleted"]
                avoid_actions.add(action)

            # optimize for the nearest non-depleted cluster
            obs = []

            for a in range(len(state)):
                if a not in avoid_actions:
                    dist = round(state[a], 2)
                    obs.append([dist, a])

            action = min(obs)[1]

        sum_reward += reward

        if done:
            if verbose:
                print("DONE @ step {}".format(i))

            break

    if verbose:
        print("CUMULATIVE REWARD", round(sum_reward, 3))

    return sum_reward


######################################################################
## main entry point

if __name__ == "__main__":
    dataset_path = Path.cwd() / Path("jester-data-1.csv")
    df = pd.DataFrame(JokeRec.load_data(dataset_path))

    # impute missing values with column mean (avg rating per joke)
    # https://scikit-learn.org/stable/modules/impute.html
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(df.values)
    X = imp.transform(df.values).T

    # used inertia to estimate the hyperparameter (i.e., user segmentation)
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_stability_low_dim_dense.html
    # https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
    K_CLUSTERS = 12

    # k-means clustering
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    km = KMeans(n_clusters=K_CLUSTERS)
    km.fit(X)

    # segment the items by their respective cluster labels
    CENTERS = km.cluster_centers_
    CLUSTERS = defaultdict(set)
    labels = km.labels_

    for i in range(len(labels)):
        CLUSTERS[labels[i]].add(i)


    # prepare the configuration for the custom environment
    config = {
        "env": JokeRec,
        "num_workers": 3, # set to 0 for debug
        "env_config": {
            "dataset": dataset_path,
            "dense": str(JokeRec.DENSE_SUBMATRIX),
            "clusters": repr(dict(CLUSTERS)),
            "centers": repr(CENTERS.tolist())
            }
        }

    # measure the performance of a random-action baseline
    #pdb.set_trace()
    history = []

    for _ in range(10):
        sum_reward = run_one_episode(config["env_config"], naive=True, verbose=False)
        history.append(sum_reward)

    baseline = sum(history) / len(history)
    print("\nBASELINE CUMULATIVE REWARD", round(baseline, 3))
    #sys.exit(0)


    # init directory in which to save checkpoints
    chkpt_root = "tmp/rec"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "JokeRec-v0"
    register_env(select_env, lambda config: JokeRec(config))


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_workers"] = 3	# set to 0 for debug
    config["env_config"] = {
        "dataset": dataset_path,
        "dense": str(JokeRec.DENSE_SUBMATRIX),
        "clusters": repr(dict(CLUSTERS)),
        "centers": repr(CENTERS.tolist())
        }

    agent = ppo.PPOTrainer(config, env=select_env)

    # train a policy with RLlib using PPO
    status = "{:2d}  reward {:6.2f}/{:6.2f}/{:6.2f}  len {:4.2f}  saved {}"
    n_iter = 50

    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print("\n", model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = JokeRec(config["env_config"])

    state = env.reset()
    sum_reward = 0

    for step in range(JokeRec.MAX_STEPS * 5):
        try:
            action = agent.compute_action(state)
            state, reward, done, info = env.step(action)
            sum_reward += reward

            print("reward", round(reward, 3), round(sum_reward, 3))
            env.render()
        except Exception:
            traceback.print_exc()

        if done:
            # report at the end of each episode
            print("CUMULATIVE REWARD", round(sum_reward, 3), "\n")
            state = env.reset()
            sum_reward = 0

    ray.shutdown()
