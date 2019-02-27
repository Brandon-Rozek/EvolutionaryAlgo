import random
import numpy as np
import rltorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

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
def fitness(model_dict):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = Policy(state_size, action_size)
    model.load_state_dict(model_dict)
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
    return total_reward


# make_model should be a function that returns a nn.Module
class Population:
    def __init__(self, model, population_size, fitness_fn, keep_best = 1, mutation_rate = 0.01, sigma = 0.1):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.keep_best = keep_best
        self.sigma = sigma
        assert self.sigma >= 0
        assert self.keep_best >= 0
        assert self.population_size > 0
        assert self.keep_best < self.population_size
        self.pop = self._generate_population(model, population_size)
        
        # Probability that an individual will last to the next generation
        self.survivability = np.full(shape=(population_size), fill_value = 1 / population_size)
        self.calculate_fitness = fitness_fn

    def _generate_population(self, model, population_size):
        pop = []
        for i in range(population_size):
            member = {}
            for key, value in model.state_dict().items():
                member[key] = value + self.sigma * torch.randn(*value.shape)
            pop.append(member)
        return pop
        
    def _calculate_survivability(self, pop): 
        fitness = np.array(list(map(self.calculate_fitness, pop)))
        # Make fitness non-negative
        if fitness.min() <= 0:
            fitness += (-1 * fitness.min()) + 1e-10 # Add some random constant to avoid 0 probability
        return fitness / fitness.sum()

    def _select_survivors(self, population, survivability):
        population_size = len(population)
        survivors_indices = np.random.choice(range(0, population_size), size=(population_size - self.keep_best) * 2, p=survivability)
        return [population[i] for i in survivors_indices]

    def _crossover(self, parents):
        parent_ind = np.array(range(0, len(parents)))
        parent1_ind = np.random.choice(parent_ind, size = len(parents) // 2, replace=False)
        parent2_ind = np.setdiff1d(parent_ind, parent1_ind)
        parent1 = [parents[i] for i in parent1_ind]
        parent2 = [parents[i] for i in parent1_ind]
        children = []
        for parent1, parent2 in zip(parent1, parent2):
            child = {}
            for key in parent1.keys():
                crossover_ind = random.randint(0, len(parent1[key]))
                child_value = torch.cat((parent1[key][:crossover_ind], parent2[key][crossover_ind:]))
                child_value = self._mutate(child_value)
                child[key] = child_value
                
            children.append(child)
        return children
    
    def _mutate(self, child):
        if np.random.rand() < self.mutation_rate:
            child += self.sigma * torch.randn(*child.shape)
        return child
    
    def __iter__(self):
        return self
    
    # This function is suppose to take us to the next generation
    def __next__(self):
        survivability = self._calculate_survivability(self.pop)
        if self.keep_best > 0:
            survivor_ind = np.argsort(survivability)[-self.keep_best:]
        parents = self._select_survivors(self.pop, survivability)
        children = self._crossover(parents)
        next_pop = [self.pop[i] for i in survivor_ind] + children
        self.pop = next_pop
        return next_pop

    def solution(self):
        return self.pop[self.survivability[-1]]


def test():
    p = Population(Policy(6, 3), 100, fitness)
    for i in range(100):
        next(p)
    return p.solution()