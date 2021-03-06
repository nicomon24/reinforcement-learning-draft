
# coding: utf-8

# # Policy Gradient in continuous action space using TF
# We will use TF to apply PG method to MountainCarContinuos environment

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
    def __init__(self, ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[]):

        # save inputs for copy
        self.ft = ft
        self.D = D
        self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
        self.hidden_layer_sizes_var = hidden_layer_sizes_var

        ##### model the mean #####
        self.mean_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_mean:
            layer = HiddenLayer(M1, M2)
            self.mean_layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
        self.mean_layers.append(layer)


        ##### model the variance #####
        self.var_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_var:
            layer = HiddenLayer(M1, M2)
            self.var_layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
        self.var_layers.append(layer)

        # gather params
        self.params = []
        for layer in (self.mean_layers + self.var_layers):
            self.params += layer.params

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        def get_output(layers):
            Z = self.X
            for layer in layers:
                Z = layer.forward(Z)
            return tf.reshape(Z, [-1])

        # calculate output and cost
        mean = get_output(self.mean_layers)
        var = get_output(self.var_layers) + 10e-5 # smoothing

        # log_probs = log_pdf(self.actions, mean, var)
        norm = tf.contrib.distributions.Normal(mean, var)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        # log_probs = norm.log_prob(self.actions)
        # cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*tf.log(2*np.pi*var)) + 0.1*tf.reduce_sum(mean*mean)
        # self.cost = cost
        # self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)
        # self.train_op = tf.train.AdagradOptimizer(10e-3).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(10e-5, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)

    def set_session(self, session):
        self.session = session

    def init_vars(self):
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        # print("action:", p)
        return p

    def copy(self):
        clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_mean)
        clone.set_session(self.session)
        clone.init_vars() # tf will complain if we don't do this
        clone.copy_from(self)
        return clone

    def copy_from(self, other):
        # collect all the ops
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        # now run them all
        self.session.run(ops)

    def perturb_params(self):
        ops = []
        for p in self.params:
            v = self.session.run(p)
            noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
            if np.random.random() < 0.1:
                # with probability 0.1 start completely from scratch
                op = p.assign(noise)
            else:
                op = p.assign(v + noise)
            ops.append(op)
        self.session.run(ops)


# Next we will create the play_one function

# In[5]:

def play_one(env, pmodel, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever
        # the 200 limit seems a bit early
        action = pmodel.sample_action(observation)
        # oddly, the mountain car environment requires the action to be in
        # an object where the actual action is stored in object[0]
        observation, reward, done, info = env.step([action])

        totalreward += reward
        iters += 1
    return totalreward


def play_multiple_episodes(env, T, pmodel, gamma, print_iters=False, status=False):
    totalrewards = np.empty(T)
    r = range(T)
    if status:
        r = tqdm_notebook(range(T), desc='Episodes'):
    for i in range(T):
        totalrewards[i] = play_one(env, pmodel, gamma)
        if print_iters:
            print(i, "avg so far:", totalrewards[:(i+1)].mean())

    avg_totalrewards = totalrewards.mean()
    print("avg totalrewards:", avg_totalrewards)
    return avg_totalrewards


# We create the random search function to optimize parameters

# In[6]:

def random_search(env, pmodel, gamma):
    totalrewards = []
    best_avg_totalreward = float('-inf')
    best_pmodel = pmodel
    num_episodes_per_param_test = 3
    for t in tqdm_notebook(range(100), desc='HillClimb'):
        tmp_pmodel = best_pmodel.copy()
        tmp_pmodel.perturb_params()

        avg_totalrewards = play_multiple_episodes(
          env,
          num_episodes_per_param_test,
          tmp_pmodel,
          gamma
        )
        totalrewards.append(avg_totalrewards)

        if avg_totalrewards > best_avg_totalreward:
            best_pmodel = tmp_pmodel
    return totalrewards, best_pmodel


# In[7]:

ft = FeatureTransformer(env, n_components=100)
D = ft.dimensions
pmodel = PolicyModel(ft, D, [], [])
session = tf.InteractiveSession()
pmodel.set_session(session)
pmodel.init_vars()
gamma = 0.99

totalrewards, pmodel = random_search(env, pmodel, gamma)

avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma, status=True)

plt.plot(totalrewards)
plt.show()

