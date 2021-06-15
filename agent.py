import random

import gym
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, s, q, actions, epsilon, alpha, gamma, e_min, e_rate):
        self.s = s
        self.current_epsilon = epsilon
        self.q = q
        self.rewards = []
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.e_min = e_min
        self.e_rate = e_rate

    def play_all_games(self, episodes):
        game_counter = 0
        self.env = gym.make('FrozenLake-v0')
        while game_counter < episodes:
            self.env.reset()
            self.rewards.append(self.play_one_game())
            game_counter += 1
            # if game_counter % 10_000 == 0:
            #     print(f"Doing good: {game_counter}")
        average_scores = self.plot_average_score()
        average_score = average_scores[-100:].mean() 
        print(f"Score: {average_score}")
        return average_score

    def plot_average_score(self):
        average_scores = np.asarray(self.rewards).reshape((-1, 100)).mean(axis=-1)
        _, ax = plt.subplots()
        ax.plot(average_scores)
        return average_scores

    def play_one_game(self):
        game_ended = False
        rewards = 0
        while not game_ended:
            self.choose_action()
            s_prime, reward, game_ended = self.move()
            rewards += reward
            self.update_q(s_prime, reward)
            self.s = s_prime
            self.decrease_epsilon()
        return rewards

    def choose_action(self):
        rand = random.random()
        if rand < self.current_epsilon:
            self.a = random.choice(self.actions)
        else:
            a_max = np.argmax(self.q[self.s])
            self.a = self.actions[a_max]

    def move(self):
        s_prime, reward, game_ended, _ = self.env.step(self.a)
        return s_prime, reward, game_ended

    def update_q(self, s_prime, reward):
        current_action = self.q[self.s][self.a]
        a_max = np.argmax(self.q[self.s])
        newq = (
                current_action +
                self.alpha * (
                    reward +
                        self.gamma*(self.q[s_prime][a_max])
                        - current_action
                    )
            )
        self.q[self.s][self.a] = newq

    def decrease_epsilon(self):
        if self.current_epsilon > self.e_min:
            self.current_epsilon *= self.e_rate

