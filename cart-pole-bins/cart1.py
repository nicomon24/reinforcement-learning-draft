# First attempt at cart-pole, using bins and MC with eps-greedy

import gym, time
import numpy as np

env = gym.make('CartPole-v0')

# Cartpole has 4 state vars
# 1 action, can be 0 or 1

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = [0,1]

def random_action(a, eps=0.1):
    p = np.random.random()
    if p < (1-eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def hash_float(f):
    if int(f) <= -10:
        return 0
    elif int(f) >= 10:
        return 20
    else:
        return int(f) + 10

def hash_state(observation):
    return ''.join([str(hash_float(x)) for x in observation])

def play(env, policy, render=False):
    observation = env.reset()
    state = hash_state(observation)
    action = random_action(policy[state])
    states_actions_rewards = [(state, action, 0)]
    # Play the game greedily
    done = False
    while not done:
        if render:
            env.render()
        observation, reward, done, info = env.step(action)
        state = hash_state(observation)
        if done:
            states_actions_rewards.append((state, None, reward))
            break
        else:
            action = random_action(policy[state])
            states_actions_rewards.append((state, action, reward))
    # Return rewards
    G = 0
    states_actions_returns = []
    first = True
    for state, action, reward in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((state, action, G))
        G = reward + GAMMA * G
    states_actions_returns.reverse() # Reverse because we want first-visit MC
    return states_actions_returns


def all_states():
    all_states = []
    for i in range(21):
        for j in range(21):
            for k in range(21):
                for z in range(21):
                    all_states.append(str(i)+str(j)+str(k)+str(z))
    return all_states

def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

for i_episode in range(1):
    # Random initial policy
    policy = {}
    for s in all_states():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print("Random initial policy")
    # Init Q
    Q = {}
    returns = {}
    states = all_states()
    # Init Q to avoid error on argmax of empty dict
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
            returns[(s,a)] = []
    # Start main loop
    c = 0
    while True:
        c += 1
        biggest_change = 0
        # Play game
        states_actions_returns = play(env, policy, render=(c > 1000))
        # Update policy
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            sa = (s,a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)

        # New policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
