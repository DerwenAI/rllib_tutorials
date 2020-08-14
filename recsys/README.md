# RLlib Tutorial: A recommender system based on reinforcement learning

This codes illustrates one way to build a recommender system based on
reinforcement learning, using [Ray RLlib](https://rllib.io/) used for
[Anyscale Academy](https://github.com/anyscale/academy).

The data comes from the [Jester](https://goldberg.berkeley.edu/jester-data/)
collaborative filtering dataset for an online joke rating system.
For further details about that research project, see:

[Eigentaste: A Constant Time Collaborative Filtering Algorithm](http://www.ieor.berkeley.edu/~goldberg/pubs/eigentaste.pdf)
Ken Goldberg, Theresa Roeder, Dhruv Gupta, Chris Perkins.
*Information Retrieval*, 4(2), 133-151. (July 2001)


### Installation

```
pip install -r requirements.txt
```


### Running

Source code for the example recommender system is in `recsys.py` and
to run it with minimal settings (to exercise the code) use:

```
python recsys.py
```

To see the available command line options use:

```
python recsys.py --help
```

A full run takes about 5-10 minutes on a recent model MacBook Pro
laptop.


### Troubleshooting

If you are reviewing a branch of this repo to evaluate a reported
issue, the latest exception trace from the Ray worker (if any) will be
located in `error.txt` in this directory.


### Analysis

There is a Jupyter notebook `cluster.ipynb` which was used to
determine how to optimize the clustering used by the recommender
system.


### Dependencies

This code was developed in `Python 3.7.4` on `macOS 0.13.6 (17G14019)`
using the following releases of library dependencies related to
[Ray](https://ray.io/):

```
gym==0.17.2
numpy==1.18.5
pandas==1.0.5
ray==0.8.6
scikit-learn==0.21.3
tensorboard-plugin-wit==1.7.0
tensorboard==2.3.0
tensorboardX==1.9
tensorflow-estimator==2.3.0
tensorflow-probability==0.9.0
tensorflow==2.3.0
```