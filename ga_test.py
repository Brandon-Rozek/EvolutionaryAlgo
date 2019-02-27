import random
import numpy as np


# Let's solve the function f(x, y) = -2x^2 - 3(y - 4)^2
def fitness(x):
    return -2 * (x[:, 0] ** 2) - 3 * (x[:, 1] - 4)**2

class Population:
    def __init__(self, output_size, population_size, fitness_fn, low = 0., high = 1., keep_best = 1, mutation_rate = 0.001):
        self.population_size = population_size
        self.output_size = output_size
        self.low = low
        self.high = high
        self.mutation_rate = mutation_rate
        self.keep_best = keep_best
        assert self.keep_best >= 0
        assert self.population_size > 0
        assert self.keep_best < self.population_size
        self.pop = self._generate_population(output_size, population_size, low = low, high = high)
        
        # Probability that an individual will last to the next generation
        self.survivability = np.full(shape=(population_size), fill_value = 1 / population_size)
        self.calculate_fitness = fitness_fn

    def _generate_population(self, output_size, population_size, low = 0., high = 1.):
        return np.random.uniform(low, high, size=(population_size, output_size))
        
    def _calculate_survivability(self, pop):
        fitness = self.calculate_fitness(pop)
        # Make fitness non-negative
        if fitness.min() <= 0:
            fitness += (-1 * fitness.min()) + np.finfo('float').eps
        return fitness / fitness.sum()

    def _select_survivors(self, population, survivability):
        population_size = len(population)
        survivors_indices = np.random.choice(range(0, population_size), size=(population_size - self.keep_best) * 2, p=survivability)
        return population.take(survivors_indices, axis = 0)

    def _crossover(self, parents):
        parent_ind = np.array(range(0, len(parents)))
        parent1_ind = np.random.choice(parent_ind, size = len(parents) // 2, replace=False)
        parent2_ind = np.setdiff1d(parent_ind, parent1_ind)
        parents1 = parents[parent1_ind]
        parents2 = parents[parent2_ind]
        children = []
        for parent1, parent2 in zip(parents1, parents2):
            crossover_ind = random.randint(0, self.output_size)
            child = np.zeros_like(parent1)
            child[:crossover_ind] = parent1[:crossover_ind]
            child[crossover_ind:] = parent2[crossover_ind:]
            child = self._mutate(child)
            children.append(child)
        return np.vstack(children)
    
    def _mutate(self, child):
        for i in range(len(child)):
            if np.random.rand() < self.mutation_rate:
                child[i] = np.random.uniform(self.low, self.high)
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
        next_pop = np.concatenate((self.pop.take(survivor_ind, axis = 0), children))
        self.pop = next_pop
        return next_pop

    def solution(self):
        return self.pop.take(sorted(self.survivability)[-1], axis = 0)


def test():
    p = Population(2, 100, fitness, low = -10, high = 10)
    for i in range(10000):
        next(p)
    return p.solution()