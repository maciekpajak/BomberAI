import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from tqdm import tqdm

from src.game import Algorithm, Action
from src.qlearning.qmodel import QModel


def get_reward(player_alive, action, is_move_possible, suicide, kills, destroyed_boxes):
    r = -0.1
    if action == Action.NO_ACTION:
        r -= 3
    if not is_move_possible:
        r -= 5
    if not player_alive and not suicide:
        r -= 50
    if not player_alive and suicide:
        r -= 1000
    r += kills * 200
    r += destroyed_boxes * 10
    return r


if __name__ == "__main__":
    model = QModel()
    epsilon = 0.3
    de = 0.01
    discount = 0.98
    lr = 0.2
    gamma = 0.99
    n_past = 200
    epochs = 30
    episodes = 500
    model.compile(get_reward=get_reward,
                  learning_rate=lr,
                  discount=discount,
                  epsilon=epsilon,
                  de=de,
                  gamma=gamma,
                  n_past_states=n_past,
                  state_type='circle',
                  state_range=3,
                  min_enemy_dist=10)

    grid_path = Path('.') / 'maps' / 'standard' / 'M.csv'
    grid = np.genfromtxt(grid_path, delimiter=',')
    en1_alg = Algorithm.DFS
    en2_alg = Algorithm.WANDER
    en3_alg = Algorithm.RANDOM
    model.set_game(grid=grid,
                   en1_alg=en1_alg,
                   en2_alg=en2_alg,
                   en3_alg=en3_alg,
                   box_density=(3, 6),
                   shuffle_positions=True,
                   max_playing_time=120)

    history = model.fit(epochs=epochs,
                        episodes=episodes,
                        start_epoch=0,
                        show_game=True,
                        path_to_save='qtables/tests/qtable.csv',
                        log_file='qtables/tests/log.csv')
