# RLlib Tutorials

These _reinforcement learning_ tutorials use environments from 
[OpenAI Gym](https://gym.openai.com/) to illustrate how to train policies 
in [RLlib](https://ray.readthedocs.io/en/latest/rllib.html).


## Getting Started

To get started use `git` to clone this public repository:
```
git clone https://github.com/DerwenAI/rllib_tutorials.git
cd rllib_tutorials
```

Then use `pip` to install the required dependencies:
```
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Alternatively, if you use `conda` for installing Python packages:
```
conda create -n rllib_tutorials python=3.7
conda activate rllib_tutorials
python3 -m pip install -r requirements.txt
```

Use [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) to run the
notebooks.
Connect into the directory for this repo, then launch JupyterLab with the
command line:

```
jupyter-lab
```

---

## Tutorial: Intro to Reinforcement Learning and Tour Through RLlib

*Intro to Reinforcement Learning and Tour Through RLlib* covers an
introductory, hands-on coding tour through RLlib and related
components of Ray used for reinforcement learning applications in
Python.
This webinar begins with a lecture that introduces reinforcement
learning, including the essential concepts and terminology, plus show
typical coding patterns used in RLlib.
We'll also explore four different well-known reinforcement learning
environments through hands-on coding.
The intention is to compare and contrast across these environments to
highlight the practices used in RLlib.
Then we'll follow with Q&A.

### Prerequisites

  * some Python programming experience
  * some familiarity with machine learning
  * clone/install the Git repo
  * no previous work in reinforcement learning
  * no previous hands-on experience with RLlib

### Background

See also:

  * [*Intro to RLlib: Example Environments*](https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70)


## Tutorial: Using Reinforcement Learning: Custom Environments, Multi-Armed Bandits, Recommendation Systems 

*Using Reinforcement Learning* begins with a brief tutorial about how to
build custom Gym environments to use with RLlib, to use as a starting point.
We’ll then explore hands-on coding for RL through two use cases:

  1. Contextual bandits with a financial portfolio optimization example–a real-world problem addressed with a “constrained” class of RL algorithms
  2. Building a recommender system with RLlib–new approaches to recommenders, which can be adapted to similar use cases

### Prerequisites

  * Some Python programming experience
  * Some familiarity with machine learning
  * Clone/install the Git repo
  * *Intro to Reinforcement Learning and Tour Through RLlib* or equivalent

---

## Resources

  * [Forum](https://discuss.ray.io/)
  * [GitHub](https://github.com/ray-project/ray)
  * [Twitter](https://twitter.com/raydistributed)
  * [Slack](https://tinyurl.com/rayslack)

**Ray Summit**  
June 22-24, 2021  
online, free registration  
<https://www.anyscale.com/ray-summit-2021>
