
# coding: utf-8

# # Mountain Car with RBF

# First we import the graphic libs and we define a function to display the run

# In[1]:

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tnrange, tqdm_notebook
import numpy as np

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


# Now we create the environment (Mountain car V0)

# In[2]:

import gym
env = gym.make("MountainCar-v0")


# Next we need to create the RBF network from sklearn lib. The first part will be the feature transformer, that transforms feature as a kernel approximator. We will use RBF with different gammas to have more granularity.

# In[3]:

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

class FeatureTransformer:
    
    def __init__(self, env, n_components=100):
        # First we need to sample from the observation_space to tune the RBFs (10k samples)
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        # Define the scaler
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        # Setup RBFsampler to transform the observation space
        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=n_components)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=n_components)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=n_components))
        ])
        # WHAT IS DIS (maybe used for dimensions)
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))
        # Save all
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


# Next we want to create an StochasticGradientDescend regressor, that combined with the feature transformer, will lead to our implementation of the RBF

# In[4]:

# SGDRegressor we want
class Model:
    
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.feature_transformer = feature_transformer
        # Models contains a model for every possible action
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform( [env.reset()] ), [0])
            self.models.append(model)
        
    # Predict Q for current state (after transforming it)
    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([m.predict(X)[0] for m in self.models])
    
    # Update the RBF network with the estimated Q
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        self.models[a].partial_fit(X, [G])

    # Take a possible random action, based on epsilon (or the best one, max Q)
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
  


# Then, we want to create a functions that executes one episode and update the model we pass to it 

# In[5]:

# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, render=False):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    frames = []
    while not done and iters < 10000:
        if render:
            frames.append(env.render(mode = 'rgb_array'))
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)
        # update the model
        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G)
        if reward == 1: # if we changed the reward to -200
            totalreward += reward
        iters += 1

    return totalreward, frames


# In this final part, we instantiate everything we need and run the environment

# In[6]:

ft = FeatureTransformer(env)
model = Model(env, ft, "constant")
# learning_rate = 10e-5
# eps = 1.0
gamma = 0.99

N_EPISODES = 300
totalrewards = np.empty(N_EPISODES)
for episode in tqdm_notebook(range(N_EPISODES), desc='Episodes'):
    # eps = 1.0/(0.1*n+1)
    eps = 0.1*(0.97**episode)
    # eps = 0.5/np.sqrt(n+1)
    totalreward, _ = play_one(model, eps, gamma)
    totalrewards[episode] = totalreward

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()


'''
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier()   
clf.fit(X_features, y)
print(clf.score(X_features, y))

observation = env.reset()
done = False
frames = []
while not done:
    observation, reward, done, _ = env.step(env.action_space.sample())
    frames.append(env.render(mode = 'rgb_array'))
    
# Finisssssh
env.render(close=True)
display_frames_as_gif(frames)
'''


# Now we want to run the environment with the actions specified by the policy and display it

# In[7]:

_, frames = play_one(model, 0, 0.99, render=True)
env.render(close=True)
print(len(frames))
display_frames_as_gif(frames)


# In[ ]:



