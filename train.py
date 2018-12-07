# Imports from model.py to get the model
# Runs model.py, do Bellman update equation
# Backpropagate on loss
import matplotlib.pyplot as plt
import numpy as np
import params as P
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from IPython.display import clear_output
from model import DQN
from tqdm import tqdm_notebook
from utils import Replay, LinearSchedule
from wrappers import make_atari, wrap_deepmind

def plot(step, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (step, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

def init_weights(model):
    for param in model.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param)

def train_dqn(env_name):
    env = wrap_deepmind(make_atari(env_name))
    num_actions = env.action_space.n
    print('Num actions: {}'.format(num_actions))
    model = DQN(out_size=num_actions)
    target_model = DQN(out_size=num_actions)
    criterion = nn.SmoothL1Loss()
    print('Created models')

    cuda = False
    if torch.cuda.is_available():
        cuda = True
        model = model.cuda()
        target_model = target_model.cuda()
        print('GPU: {}'.format(torch.cuda.get_device_name(0)))

    model.apply(init_weights)
    target_model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())#, lr=0.00001)
    print('Initalized models')

    schedule = LinearSchedule(P.start_eps, P.end_eps, P.steps_eps)
    replay = Replay(P.replay_size, P.batch_size, cuda)
    state = env.reset()
    num_updates = 0
    eps_reward = 0
    rewards = []
    losses = []
    # populate replay with random policy
    print('Populating replay')
    for i in tqdm_notebook(range(P.replay_start_size), desc='Populating replay'):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        replay.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
    print('Starting training')
    state = env.reset()
    for i in tqdm_notebook(range(P.num_steps), desc='Total steps'):
        if schedule.choose_random():
            action = env.action_space.sample()
        else:
            model_input = torch.from_numpy(np.array(state)[None, :]).type(torch.FloatTensor)
            if cuda:
                model_input = model_input.cuda()
            q_values = model(model_input)
            action = int(q_values.argmax(1)[0])
        next_state, reward, done, _ = env.step(action)
        eps_reward += reward
        replay.add(state, action, reward, next_state, done)
        state = next_state
        if i % P.update_freq == 0:
            loss = compute_loss(replay, optimizer, model, target_model, P.gamma, criterion)
            num_updates += 1
            if num_updates % P.target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
        if done:
            rewards.append(eps_reward)
            losses.append(loss.item())
            eps_reward = 0
            state = env.reset()
        if i % P.plot_every == 0 and i > 0:
            plot(i, rewards, losses)
        if i % P.save_every == 0:
            torch.save(model, 'experiment/{}_model'.format(i))
            pickle.dump(losses, open("experiment/{}_losses.p".format(i), "wb"))
            pickle.dump(rewards, open("experiment/{}_rewards.p".format(i), "wb"))

def compute_loss(replay, optimizer, model, target_model, gamma, criterion):
    states, actions, rewards, next_states, dones = replay.sample_tensor()
    next_states = next_states
    model_q = model(states) # (batch, actions)
    model_qa = torch.gather(model_q, 1, actions[:, None]).squeeze()
    next_q = target_model(next_states).detach()
    max_next_q = next_q.max(1)[0] * (1 - dones) # (batch,)
    q = rewards + gamma * max_next_q
    loss = criterion(model_qa, q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
