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
## Gym environment

class JokeRec (gym.Env):
    DENSE_SUBMATRIX = [ 5, 7, 8, 13, 15, 16, 17, 18, 19, 20 ]
    ROW_LENGTH = 100
    MAX_STEPS = ROW_LENGTH - len(DENSE_SUBMATRIX)

    NO_RATING = "99"
    MAX_RATING = 10.0
    MAX_OBS = np.sqrt(MAX_STEPS)

    REWARD_UNRATED = -0.1	# unknown
    REWARD_DEPLETED = -0.05	# items depleted


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

        # select a random user to simulate
        self.dataset = self.load_data(config["dataset"])
        self.data_row = random.choice(self.dataset)


    def _warm_start (self):
        """
        attempt a warm start for the rec sys, by sampling
        half of the dense submatrix of most-rated items
        """
        sample_size = round(len(self.dense) / 2)

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
        #print(">> ", self.depleted)
        print(">> ", self.used)
        print(">> ", self.history)


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
    history = []

    for _ in range(10):
        sum_reward = run_one_episode(config["env_config"], naive=True, verbose=False)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nBASELINE CUMULATIVE REWARD", round(avg_sum_reward, 3))
    #sys.exit(0)

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
        check_learning_achieved(results, 3.0) #stop["episode_reward_mean"]
    except Exception:
        traceback.print_exc()

    ray.shutdown()
