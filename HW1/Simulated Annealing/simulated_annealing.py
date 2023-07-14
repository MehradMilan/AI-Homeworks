import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import random

class SimulatedAnnealing:

    def __init__(self):
        pass

    def random_state_generator(self, n: int) -> np.ndarray:
        return np.array([random.randint(0, 2) for _ in range(n)])

    def objective_function(self, state: np.ndarray, S: np.ndarray) -> int:
        return np.dot(S, state)

    def is_feasible(self, state: np.ndarray, S: np.ndarray, T: int) -> bool:
        return self.objective_function(state, S) <= T

    def flip_random_bit(self, state: np.ndarray) -> np.ndarray:
        pivot = int(random.randint(0, len(state)))
        state[pivot] = int(not(state[pivot]))
        return state

    def neighbor_state_generator(self, state: np.ndarray, S: np.ndarray, T: int, prob: float = 0.5) -> np.ndarray:
        neighbor_state = state.copy()
        while True:
            neighbor_state = self.flip_random_bit(neighbor_state)
            r = random.rand()
            if r <= prob:
                neighbor_state = self.flip_random_bit(neighbor_state)
            if self.is_feasible(neighbor_state, S, T):
                return neighbor_state            

    def cost_function(self, state: np.ndarray, S: np.ndarray, T: int) -> int:
        sum = self.objective_function(state, S)
        return np.absolute(sum - T)

    def prob_accept(self, state_cost: int, next_state_cost: int, temperature: float) -> float:
        P = np.exp(-((next_state_cost - state_cost) ** 1.48) / temperature * 1.48)
        return P


    def run_algorithm(self, S: np.ndarray, T: int, neigbor_prob: float = 0.5, stopping_iter: int = 3000, temperature: float = 30):
        state = self.random_state_generator(len(S))
        best_cost = self.cost_function(state, S, T)
        best_solution = state
        records = []

        decay_rate = temperature / stopping_iter

        for i in tqdm(range(stopping_iter)):
            neighbor_state = self.neighbor_state_generator(state, S, T, neigbor_prob)
            state_cost = self.cost_function(state, S, T)
            neighbor_state_cost = self.cost_function(neighbor_state, S, T)
            if neighbor_state_cost <= state_cost:
                state = neighbor_state
                best_cost = self.cost_function(state, S, T)
                best_solution = state
            else:
                r = random.rand()
                if r <= self.prob_accept(state_cost, neighbor_state_cost, temperature):
                    state = neighbor_state
            temperature = temperature - decay_rate
            records.append({'iteration': i, 'best_cost': best_cost,
                           'best_solution': best_solution})  # DO NOT REMOVE THIS LINE

        records = pd.DataFrame(records)  # DO NOT REMOVE THIS LINE
        return best_cost, best_solution, records
