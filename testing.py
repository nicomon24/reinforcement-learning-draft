'''
    Testing file for openAI gym
'''
import gym
from IPython import embed

env = gym.make("FrozenLake-v0")

done = False
observation = env.reset()
env.render()
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    print(observation, reward, done, info)
