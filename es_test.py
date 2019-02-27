import random
import numpy as np


# Let's solve the function f(x, y) = -2x^2 - 3(y - 4)^2
def fitness(x):
    return -2 * (x[:, 0] ** 2) - 3 * (x[:, 1] - 4)**2

class Population:
    def __init__(self, initial_guess, population_size, fitness_fn, learning_rate = 1e-4, sigma = 0.1):
        self.current_solution = initial_guess
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        assert self.population_size > 0
        assert self.sigma >= 0
        self.calculate_fitness = fitness_fn

        
    def __iter__(self):
        return self
    
    # This function is suppose to take us to the next generation
    def __next__(self):
        white_noise = np.random.randn(self.population_size, *self.current_solution.shape)
        noise = self.sigma * white_noise
        candidate_solutions = self.current_solution + noise
        fitness_values = self.calculate_fitness(candidate_solutions)
        # Mean shift and scale
        fitness_values = (fitness_values - np.mean(fitness_values)) / (np.std(fitness_values) + np.finfo('float').eps)
        new_solution = self.current_solution + self.learning_rate * np.mean(white_noise.T * fitness_values, axis = 1) / self.sigma
        self.current_solution = new_solution
        return new_solution

    def item(self):
        return self.current_solution


def test():
    guess = np.random.randn(2)
    p = Population(guess, 100, fitness)
    for i in range(10000):
        next(p)
    return p.item()