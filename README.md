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
