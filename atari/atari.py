
# coding: utf-8

# # DQN on atari's TBD
# We are going to implement DQN on the atari game TBD.

# In[20]:

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tnrange, tqdm_notebook
import numpy as np
from collections import deque

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
    display(display_animation(anim, default_mode='loop'))


# ## Load environment

# In[21]:

import gym
from gym import wrappers
# Create env
env = gym.make("Breakout-v0")
# Setup wrapper
outdir = 'tmp/dqn-results'
env = wrappers.Monitor(env, directory=outdir, force=True)
# Set seed for replication 
env.seed(42)


# ## Image preprocessing
# We are going to scale and crop the image to make our learning easier (without losing information)

# In[22]:

from scipy.misc import imresize

IM_WIDTH = 80
IM_HEIGHT = 80

def preprocess_screen(observation):
    # Crop image
    observation = observation[31:194]
    # Apply grayscale
    observation = observation.mean(axis=2)
    # Normalize 0-1
    observation = observation / 255
    # Scale to 80x80
    observation = imresize(observation, size=(IM_HEIGHT, IM_WIDTH), interp='nearest')
    return observation

obs = env.reset()
plt.imshow(preprocess_screen(obs), cmap='gray')


# Next we will create the DQN model using layers this time

# In[18]:

class DQN:
    
    def __init__(self, K, gamma, scope):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope):
            # Placeholders
            self.X = tf.placeholder(tf.float32, shape=(None, IM_HEIGHT, IM_WIDTH, 4), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            # Convolution
            self.c1 = tf.contrib.layers.conv2d(inputs=self.X, num_outputs=16, kernel_size=8, stride=4)
            self.c2 = tf.contrib.layers.conv2d(inputs=self.X, num_outputs=32, kernel_size=4, stride=2)
            self.flatted = tf.contrib.layers.flatten(inputs=self.c2)
            # Fully connected
            self.d1 = tf.contrib.layers.fully_connected(inputs=self.flatted, num_outputs=256)
            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(inputs=self.d1, num_outputs=K, activation_fn=None)
            # Cost and train
            selected_action_values = tf.reduce_sum(
                self.output * tf.one_hot(self.actions, K),
                reduction_indices=[1]
            )
            cost = tf.reduce_mean(tf.square(self.G - selected_action_values))
            self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
            self.cost = cost
        
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)

        self.session.run(ops)

    def set_session(self, session):
        self.session = session

    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict={self.X: states})
    
    def update(self, states, actions, targets):
        c, _ = self.session.run([self.cost, self.train_op],
              feed_dict={
                self.X: states,
                self.G: targets,
                self.actions: actions
              })
        return c

    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])


# We are going to use a queue to store the last 4 frames as the state 

# In[29]:

class StateQueue:
    
    def __init__(self, initial_state, n = 4):
        self.states = deque([initial_state] * n, maxlen = n)
        
    def add_state(self, state):
        self.states.append(state)
        
    def get_state(self):
        return np.transpose(np.array(self.states), [1, 2, 0])


# In[27]:

def play_one(env, experience_replay_buffer, model, tmodel, gamma, batch_size, eps, d_eps, min_eps):
    obs = env.reset()


# In[30]:

s = StateQueue(np.zeros((3,3)))
print(s.get_state().shape)


# In[ ]:



