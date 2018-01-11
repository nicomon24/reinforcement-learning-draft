'''
    Q-Learning agent
'''
from agent import Agent
from collections import defaultdict
import numpy as np
import math

ALPHA = 0.1
GAMMA = 0.9

class QAgent(Agent):

    def __init__(self, env):
        Agent.__init__(self, env)
        # Init Q
        self.Q = defaultdict(lambda: defaultdict(float))
        self.exploring = False

    def play_action(self, action, render=False, log=False):
        s = self.state['observation']
        Agent.play_action(self, action, render, log)
        s_prime = self.state['observation']
        # Get the max of Q[s'][a] for every a
        aqmax = self.max_q_for_state(s_prime)[1]
        # Update Q
        self.Q[s][action] += ALPHA * (self.state['reward'] + GAMMA * aqmax - self.Q[s][action])
        return self.state['reward']

    def choose_action(self, epsilon = 0.1):
        a = self.max_q_for_state(self.state['observation'])[0]
        # Choose randomly
        epsilon = epsilon / math.log(self.time / 10 + 1)
        p = np.random.random()
        if p >= (1 - epsilon) or a is None:
            return self.env.action_space.sample()
        else:
            return a

    # Return a tuple (action, value) as
    def max_q_for_state(self, state):
        v = list(self.Q[state].values())
        if len(v) <= 0:
            return (None, 0)
        k = list(self.Q[state].keys())
        return (k[v.index(max(v))], max(v))
