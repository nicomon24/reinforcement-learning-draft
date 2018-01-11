
# coding: utf-8

# # Policy Gradient in continuous action space using TF
# We will use TF to apply PG method to MountainCarContinuos environment. This time we will use gradient ascent instead of Hill Climb random search for the parameters

# In[1]:

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tnrange, tqdm_notebook
import numpy as np
from collections import deque
from q_learning import plot_running_avg, FeatureTransformer

get_ipython().magic('matplotlib inline')

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='once'))


# Now we create the environment (CartPole V0)

# In[2]:

import gym
env = gym.make("MountainCarContinuous-v0").env


# Then we create a layer of a network

# In[3]:

import tensorflow as tf

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
        if zeros:
            W = np.zeros((M1, M2)).astype(np.float32)
            self.W = tf.Variable(W)
        else:
            self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
            
        self.params = [self.W]
        
        self.use_bias = use_bias
        
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


# Next thing we create the policy model. This time we will have 2 outputs, the mean and the var of the distribution

# In[4]:

class PolicyModel:
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        self.ft = ft

        ##### hidden layers #####
        M1 = D
        self.hidden_layers = []
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2

        # final layer mean
        self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)

        # final layer variance
        self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        # get final hidden layer
        Z = self.X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)

        # calculate output and cost
        mean = self.mean_layer.forward(Z)
        stdv = self.stdv_layer.forward(Z) + 1e-5 # smoothing

        # make them 1-D
        mean = tf.reshape(mean, [-1])
        stdv = tf.reshape(stdv, [-1]) 

        norm = tf.contrib.distributions.Normal(mean, stdv)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        log_probs = norm.log_prob(self.actions)
        cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)

        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(
          self.train_op,
          feed_dict={
            self.X: X,
            self.actions: actions,
            self.advantages: advantages,
          }
        )

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return p


# Given that we are not using Hill Climb anymore, we also need to model the Value function

# In[5]:

class ValueModel:
    def __init__(self, D, ft, hidden_layer_sizes=[]):
        self.ft = ft
        self.costs = []

        # create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1]) # the output
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.cost = cost
        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
        cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
        self.costs.append(cost)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})        


# Next we will create the play_one function

# In[6]:

def play_one_td(env, pmodel, vmodel, gamma, render = False):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    frames = []

    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        if render:
            frames.append(env.render(mode = 'rgb_array'))
        
        action = pmodel.sample_action(observation)
        prev_observation = observation
        # oddly, the mountain car environment requires the action to be in
        # an object where the actual action is stored in object[0]
        observation, reward, done, info = env.step([action])
        totalreward += reward
        # Update models
        V_next = vmodel.predict(observation)
        G = reward + gamma * V_next
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)
        iters += 1
    return totalreward, iters


# In[7]:

ft = FeatureTransformer(env, n_components=100)
D = ft.dimensions
pmodel = PolicyModel(D, ft, [])
vmodel = ValueModel(D, ft, [])
init = tf.global_variables_initializer()
session = tf.InteractiveSession()
session.run(init)
pmodel.set_session(session)
vmodel.set_session(session)
gamma = 0.95

N = 50
totalrewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    totalreward, num_steps = play_one_td(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 1 == 0:
        print("episode:", n, "total reward: %.1f" % totalreward, "num steps: %d" % num_steps, "avg reward (last 100): %.1f" % totalrewards[max(0, n-100):(n+1)].mean())

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()

