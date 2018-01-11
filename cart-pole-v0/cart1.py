# First attempt at cart-pole

import gym, time

env = gym.make('CartPole-v0')

for i_episode in range(1):
    observation = env.reset()
    env.render()
    done = False
    action = 1
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        print(observation, reward, done, info)
        time.sleep(1)
