# Simple Implementation of Deep Q from Demonstrations Algorithm

- Basic implementation of Deep Q-learning from Demonstrations with Inaccurate Feedback [paper](https://arxiv.org/pdf/1704.03732.pdf).

- Applied to a custom Breakout game environment

## Rewards

- 1 for each brick broken, -10 for dying, 0 otherwise

## Extra Reward (for faster convergence)

- 1 if ball is aligned with the paddle (ie. if ball were to fall down vertically, the paddle would catch it), -1 otherwise

## Files

- main.py -> main training algorithm, change parameters as you wish.

- Agent.py -> Contains the memory, the networks, performs the learning update.

- Policy.py -> Network(s).

- prioritized_memory.py -> List of transitions (demonstrations/exploration) and respective TD-Errors for prioritized sampling

- UsePolicy.py -> Watch a policy play the game.

## Acknowledgements:
- Loss functions adapted from [here](https://github.com/go2sea/DQfD)
