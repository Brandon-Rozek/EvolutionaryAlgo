import random
import numpy as np
import rltorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
from copy import deepcopy

class Policy(nn.Module):
  def __init__(self, state_size, action_size):
    super(Policy, self).__init__()
    self.state_size = state_size
    self.action_size = action_size

    self.fc1 = nn.Linear(state_size, 125)
    self.fc_norm = nn.LayerNorm(125)
    
    self.fc2 = nn.Linear(125, 125)
    self.fc2_norm = nn.LayerNorm(125)

    self.action_prob = nn.Linear(125, action_size)

  def forward(self, x):
    x = F.relu(self.fc_norm(self.fc1(x)))
    x = F.relu(self.fc2_norm(self.fc2(x)))
    x = F.softmax(self.action_prob(x), dim = 1)
    return x



env = gym.make("Acrobot-v1")
def fitness(model):
    state = torch.from_numpy(env.reset()).float().unsqueeze(0)
    total_reward = 0
    done = False
    while not done:
        action_probabilities = model(state)
        distribution = Categorical(action_probabilities)
        action = distribution.sample().item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = torch.from_numpy(next_state).float().unsqueeze(0)
    return -total_reward

# make_model should be a function that returns a nn.Module
class Population:
    def __init__(self, model, population_size, fitness_fn, learning_rate = 1e-1, sigma = 0.05):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        assert self.sigma >= 0
        assert self.population_size > 0
        self.calculate_fitness = fitness_fn
    
    def __iter__(self):
        return self
    
    # This function is suppose to take us to the next generation
    def __next__(self):
        ## Generate Noise
        model_dict = self.model.state_dict()
        white_noise_dict = {}
        noise_dict = {}
        for key in model_dict.keys():
            white_noise_dict[key] = torch.randn(self.population_size, *model_dict[key].shape)
            noise_dict[key] = self.sigma * white_noise_dict[key]
        
        ## Generate candidate solutions
        candidate_solutions = []
        for i in range(self.population_size):
            candidate_statedict = {}
            for key in model_dict.keys():
                candidate_statedict[key] = model_dict[key] + noise_dict[key][i]
            candidate = Policy(self.model.state_size, self.model.action_size)
            candidate.load_state_dict(candidate_statedict)
            candidate_solutions.append(candidate)
        
        ## Calculate fitness
        fitness_values = torch.tensor([self.calculate_fitness(x) for x in candidate_solutions])
        print("Average fitness: ", fitness_values.mean())
        # Mean shift, scale
        fitness_values = (fitness_values - fitness_values.mean()) / (fitness_values.std() + np.finfo('float').eps)

        ## Insert adjustments into gradients slot
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                noise_dim_n = len(white_noise_dict[name].shape)
                dim = np.repeat(1, noise_dim_n - 1).tolist() if noise_dim_n > 0 else []
                param.grad = (white_noise_dict[name] * fitness_values.float().reshape(self.population_size, *dim)).mean(0) / self.sigma
        self.optimizer.step()

        return deepcopy(self.model)

p = Population(Policy(6, 3), 1000, fitness)

def iterate():
    for i in range(10):
        next(p)