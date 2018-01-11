'''
    This file plays the environment and calls the agent at every step.
'''
import gym
from q import QAgent
from IPython import embed
import numpy as np
from collections import deque

# Create environment
env = gym.make("FrozenLake-v0")
# Create agent
agent = QAgent(env)
# Create deque to store last results
last_r = deque(maxlen = 100)

for i in range(100000):
    r = agent.play_episode(log=False, render=False)
    last_r.append(r)
    print("Current mean:", np.mean(last_r))
