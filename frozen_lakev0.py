import gym
import matplotlib.pyplot as plt

MAX_GAMES = 1000

env = gym.make('FrozenLake-v0')
env.reset()

average_reward_per_ten_games = []
reward_per_iter = []
game_ended = False
steps_per_game = 0
actions = (1, 2)

for game_counter in range(1, MAX_GAMES + 1):
    while not game_ended:
        chosen_action = actions[steps_per_game % 2]
        tile, reward, game_ended, info = env.step(chosen_action)  # take a random action
        steps_per_game += 1
    reward_per_iter.append(reward)
    if game_counter % 10 == 0:
        average_reward_per_ten_games.append(sum(reward_per_iter) / 10)
        reward_per_iter = []
    env.close()
    env = gym.make('FrozenLake-v0')
    env.reset()
    game_ended = False
    steps_per_game = 0

plt.plot(average_reward_per_ten_games)
plt.show()
env.close()
