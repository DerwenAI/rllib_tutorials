#!/usr/bin/env python
# encoding: utf-8

import pdb
import sys
import traceback

from collections import defaultdict
from gym import spaces
from pathlib import Path
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import grid_search
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import csv
import gym
import numpy as np
import pandas as pd
import random
import ray


######################################################################
## scikit-learn clustering

def load_df (data_path):
    rows = []

    with open(data_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")

        for row in csvreader:
            conv = [None] * (len(row) - 1)

            for i in range(1, len(row)):
                if row[i] != "99":
                    conv[i - 1] = float(row[i]) / 10.0

            rows.append(conv)

    return pd.DataFrame(rows)


######################################################################
## Gym environment

def load_data (data_path):
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

    with open(data_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")

        for row in csvreader:
            conv = [None] * len(row)
            conv[0] = int(row[0])

            for i in range(1, len(row)):
                if row[i] != "99":
                    conv[i] = float(row[i]) / 10.0

            rows.append(conv)

    return rows


class JokeRec (gym.Env):
    MAX_STEPS = 10

    REWARD_UNRATED = -1.0	# unknown
    REWARD_DEPLETED = -2.0	# items depleted


    def __init__ (self, config):
        self.centers = eval(config["centers"])
        self.clusters = eval(config["clusters"])
        self.k_clusters = len(self.clusters)

        lo = np.array(np.float64(-1.0) * self.k_clusters)
        hi = np.array(np.float64(1.0) * self.k_clusters)

        self.observation_space = spaces.Box(lo, hi, dtype=np.float64)
        self.action_space = spaces.Discrete(self.k_clusters)

        # select a random user to simulate
        self.dataset = load_data(config["dataset"])
        episode_row = random.sample(self.dataset, 1)[0]
        self.data_row = episode_row[1:]


    def _get_state (self):
        state = np.sqrt(self.history)
        #assert state in self.observation_space, state
        return state


    def reset (self):
        self.count = 0
        self.used = []
        self.depleted = 0
        self.history = [ np.float64(0.0) for i in range(self.k_clusters) ]

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

        self.count += 1
        done = self.count >= self.MAX_STEPS
        info = { "item": item, "count": self.count }

        return self._get_state(), reward, done, info


    def render (self, mode="human"):
        #print(">> ", self.data_row)
        print(">> ", self.depleted)
        print(">> ", self.used)
        print(">> ", self.history)


def run_one_episode (config, verbose=False):
    env = JokeRec(config)
    env.reset()
    sum_reward = 0

    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if verbose:
            print("action:", action)
            print("obs:", i, state, reward, done)
            env.render()

        sum_reward += reward

        if done:
            if verbose:
                print("DONE @ step {}".format(i))

            break

    if verbose:
        print("CUMULATIVE REWARD", sum_reward)

    return sum_reward


######################################################################
## main entry point

if __name__ == "__main__":
    dataset_path = Path.cwd() / Path("jester-data-1.csv")
    df = load_df(dataset_path)

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
        "env_config": {
            "dataset": dataset_path,
            "clusters": repr(dict(CLUSTERS)),
            "centers": repr(CENTERS.tolist())
            }
        }

    # measure the performance of a random-action baseline
    history = []

    for _ in range(1):
        sum_reward = run_one_episode(config["env_config"], verbose=False)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nBASELINE CUMULATIVE REWARD: {:6.2}".format(avg_sum_reward))

    # train with Ray/RLlib
    stop = {
        "training_iteration": 1,
        "timesteps_total": 1000,
        "episode_reward_mean": 10.0
    }

    ray.init(ignore_reinit_error=True)
    #pdb.set_trace()

    try:
        results = tune.run("PPO", config=config, stop=stop)
        check_learning_achieved(results, 0.0) #stop["episode_reward_mean"]
    except Exception:
        traceback.print_exc()

    ray.shutdown()
