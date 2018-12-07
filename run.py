from train import train_dqn
import pickle

env_name = 'BreakoutNoFrameskip-v4'
exp_name = 'dqn'
train_dqn(env_name, exp_name, notebook=False)
