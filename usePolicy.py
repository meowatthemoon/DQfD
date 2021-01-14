from Agent import Agent
from Breakout_environment import Game
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# Initialize Environment
env = Game()

# Initialize Agent
agent = Agent(lr=0, eps=0, gamma=0, max_memory=0, n_steps=0, batch_size=0,
              tau=0, lambda_1=0, lambda_2=0, lambda_3=0, l_margin=0)

# Load policy
agent.policy.predictNet.load_state_dict(torch.load("Q_target_demo.pth"))

done = False
accumulate_rewards = 0
state = env.reset()
while not done:
    action = agent.choose_action(state)[0]

    # Step
    new_state, reward, done, feedback = env.step(action)

    # state = new state and accumulate rewards
    state = new_state
    accumulate_rewards += reward

    env.draw()
    time.sleep(0.01)

    if done == True:
        break
