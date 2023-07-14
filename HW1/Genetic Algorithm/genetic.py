from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import random


class Genetic:

    def __init__(self):
        pass

    def generate_initial_population(self, n: int, k: int) -> np.ndarray:
        return np.array([[random.randint(0, 2) for i in range(k)] for j in range(n)], dtype=int)  

    def objective_function(self, chromosome: np.ndarray, S: np.ndarray) -> int:
        return np.dot(S, chromosome)

    def is_feasible(self, chromosome: np.ndarray, S: np.ndarray, T: int) -> bool:
        return (self.objective_function(chromosome= chromosome, S=S) <= T)

    def cost_function(self, chromosome: np.ndarray, S: np.ndarray, T: int) -> int:
        cost = 0
        sum = self.objective_function(chromosome= chromosome, S=S)
        f = self.is_feasible(chromosome=chromosome, S=S, T=T)
        if f:
            cost = T - sum
        else:
            cost = sum
        return cost

    def selection(self, population: np.ndarray, S: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
        choices = []
        indices = np.random.randint(0, len(population), 4)
        for i in indices:
            choices.append(population[i])
        choices = np.array(choices)
        return (self.select_min_cost((choices[0], choices[1]), S, T), self.select_min_cost((choices[2], choices[3]), S, T))
        
    def select_min_cost(self, to_compare: Tuple[np.ndarray, np.ndarray], S: np.ndarray, T: int) -> np.ndarray:
        if(self.cost_function(to_compare[0], S, T) <= self.cost_function(to_compare[1], S, T)):
            return to_compare[0]
        else: return to_compare[1]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, S: np.ndarray, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        r = random.rand()
        if r < prob:
            l = len(parent1)
            pivot = int(random.randint(0, l))
            child1 = np.concatenate((parent1[0:pivot], parent2[pivot:l]))
            child2 = np.concatenate((parent2[0:pivot], parent1[pivot:l]))
            return (child1, child2)
        else:
            return (parent1, parent2)

    def mutation(self, child1: np.ndarray, child2: np.ndarray, prob: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        r = random.rand()
        if r < prob:
            l = len(child1)
            pivot = int(random.randint(0, l))
            m_child1 = child1
            m_child2 = child2
            m_child1[pivot] = int(not(m_child1[pivot]))
            m_child2[pivot] = int(not(m_child2[pivot]))
            return (m_child1, m_child2)
        else:
            return (child1, child2)

    def run_algorithm(self, S: np.ndarray, T: int, crossover_probability: float = 0.5, mutation_probability: float = 0.1, population_size: int = 100, num_generations: int = 100):
        best_cost = np.Inf
        best_solution = None
        records = []

        population = self.generate_initial_population(n=population_size, k=len(S))
        for i in tqdm(range(num_generations)):
            new_population = []
            while (len(new_population) < population_size):
                parents = self.selection(population, S, T)
                children = self.crossover(parents[0], parents[1], S, crossover_probability)
                children = self.mutation(children[0], children[1], mutation_probability)
                child1_cost = self.cost_function(children[0], S, T)
                child2_cost = self.cost_function(children[1], S, T)
                parent1_cost = self.cost_function(parents[0], S, T)
                parent2_cost = self.cost_function(parents[1], S, T)
                if (child1_cost < parent1_cost):
                    new_population.append(children[0])
                else:
                    new_population.append(parents[0])
                if (child2_cost < parent2_cost):
                    new_population.append(children[1])
                else:
                    new_population.append(parents[1])
            
            costs = [self.cost_function(chromosome, S, T) for chromosome in new_population]
            best_chromosome = new_population[np.argmin(costs)]

            if self.cost_function(best_chromosome, S, T) < best_cost:
                best_cost = self.cost_function(best_chromosome, S, T)
                best_solution = best_chromosome
            population = np.array(new_population)

            records.append({'iteration': i, 'best_cost': best_cost,
                        'best_solution': best_solution})  # DO NOT REMOVE THIS LINE

        records = pd.DataFrame(records)  # DO NOT REMOVE THIS LINE

        return best_cost, best_solution, records
