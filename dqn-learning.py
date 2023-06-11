import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from tqdm import tqdm

from src.game import Algorithm, Action
from src.qlearning.dqnmodel import DQNModel


def get_reward(player_alive, action, is_move_possible, suicide, kills, destroyed_boxes):
    r = -0.1
    if action == Action.NO_ACTION:
        r -= 3
    if not is_move_possible:
        r -= 5
    if not player_alive and not suicide:
        r -= 50
    if not player_alive and suicide:
        r -= 100
    r += kills * 200
    r += destroyed_boxes * 10
    return r


if __name__ == "__main__":
    model = DQNModel()
    epsilon = 0.1
    de = 0.01
    discount = 0.98
    lr = 0.1
    gamma = 0.99
    n_past = 50
    epochs = 10
    episodes = 10
    model.compile(get_reward=get_reward,
                  discount=discount,
                  epsilon=epsilon,
                  de=de,
                  state_type='full',
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
                        update_every=100,
                        batch_size=4,
                        path_to_save='qtables/dqnmodels/modelM.h5',
                        log_file='qtables/dqnmodels/log.csv')
