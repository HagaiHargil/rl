from multiprocessing import Pool
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from agent import Agent

# LEFT, DOWN, RIGHT, UP
base_line_actions = np.zeros(4, dtype='float32')
Q = {i: base_line_actions.copy() for i in range(16)}

E_MAX = [1.0]
EPISODES = 500_000
ALPHA = np.arange(0.0001, 0.03, 0.0001)
GAMMA = np.arange(0.8, 0.95, 0.01)
E_MIN = [0.01]
E_RATE = np.arange(0.9995, 0.999999995, 0.00000005)


def run_single(*params):
    agent = Agent(0, Q, np.arange(4, dtype='uint8'), E_MAX, ALPHA, GAMMA, E_MIN, E_RATE)
    score = agent.play_all_games(EPISODES)
    return (score, *params)


if __name__ == '__main__':
    params = product(E_MAX, ALPHA, GAMMA, E_MIN, E_RATE) 
    with Pool() as p:
        results = p.map(run_single, params)

