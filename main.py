from Agent import Agent
from Breakout_environment import Game
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch


def plotLearning(x, scores, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    ax.plot(x, scores, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.set_ylabel("Score", color="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def read_demonstration_file(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


# Algorithm Parameters (Authors')
k1 = 500
k2 = 400
gamma = 1.0  # todo try with 0.9 as that is what the paper said
max_memory = 100000
n_steps = 1  # todo try with 10 as that is what the paper said
tau = 10000
lambda_1 = 1.0
lambda_2 = 1.0
lambda_3 = 1e-5
l_margin = 0.8
e_d = 0.005  # bonus for demonstration # todo try with 1 as that is what the paper said
e_a = 0.001

# Environment Parameters
lr = 0.002
batch_size = 64
epsilon = 1.0  # Decreased over time
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.01

# Initialize Environment
env = Game()

# Initialize Agent
agent = Agent(lr=lr, eps=epsilon, gamma=gamma, max_memory=max_memory, n_steps=n_steps, batch_size=batch_size,
              tau=tau, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, l_margin=l_margin)

# Load Demonstration Data
demonstrations = read_demonstration_file("demonstration_20_games.data")
episode_idx = 0
start = 0

for index, transition in enumerate(demonstrations):
    s = transition[0]
    a = [transition[1]]
    r = transition[2]
    s_ = transition[3]
    done = transition[4]

    if done:
        r = 0  # todo not sure if this is necessary, just try with, without, with different value

    start += 1
    agent.store_demonstration(s, a, r, s_, done, int(episode_idx))

    if done:
        episode_idx += 1

# Pre-train
#agent.replay.tree.start = start
for i in range(k1):
    if i % 100 == 0:
        print("pretraining:", i)
    agent.learn()

# Train
accumulated_rewards_all_episodes = []
for episode in range(k2):
    s = env.reset()
    accumulated_rewards = 0
    done = False
    while not done:
        a = agent.choose_action(s)
        s_, r, done, feedback = env.step(a[0])

        accumulated_rewards += r
        r += feedback

        if done:
            r = 0  # todo not sure if this is necessary, just try with, without, with different value

        agent.store_transition(s, a, r, s_, done)
        agent.learn()
        s = s_

    # Update exploration rate
    """
    agent.eps = min_exploration_rate + \
                (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    """
    agent.eps = 0.001
    print(f"{episode}/{k2} : {accumulated_rewards}, Epsilon {agent.eps}")
    accumulated_rewards_all_episodes.append(accumulated_rewards)

# Stats
rewards_per_thousand_episodes = np.split(np.array(accumulated_rewards_all_episodes), k2 / 50)
count = 50
mean_scores = []
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 50)))
    count += 50
    mean_scores.append(sum(r / 50))

plotLearning([x for x in range(0, k2, 50)], mean_scores, "rewards_over_time_demo_no_decay_rate.jpg")
torch.save(agent.policy.targetNet.state_dict(), f"Q_target_demo.pth")
