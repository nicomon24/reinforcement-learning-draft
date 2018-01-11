'''
    Base class for an agent
'''

class Agent():

    def __init__(self, env):
        self.env = env
        self.time = 1

    def play_episode(self, render=False, log=False):
        cum_reward = 0
        self.state = {
            'observation': self.env.reset(),
            'done': False,
            'reward': 0,
            'info': {}
        }
        if render:
            self.env.render()
        while not self.state['done']:
            a = self.choose_action()
            r = self.play_action(a, render, log)
            cum_reward += r
        self.time += 1
        return cum_reward

    def play_action(self, action, render=False, log=False):
        observation, reward, done, info = self.env.step(action)
        self.state = {
            'observation': observation,
            'done': done,
            'reward': reward,
            'info': info
        }
        if render:
           self.env.render()
        if log:
            print(observation, reward, done, info)
        return reward

    def choose_action(self):
        return self.env.action_space.sample()
