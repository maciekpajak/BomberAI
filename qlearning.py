import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from tqdm import tqdm

from src.game import Algorithm, Action
from src.qlearning.qmodel import QModel


def get_reward(player_alive, action, is_move_possible, player_suicide, player_killed_enemy, sectors_cleared_by_player):
    r = 0
    if action == Action.NO_ACTION:
        r -= 3
    if not is_move_possible:
        r -= 5
    if not player_alive and not player_suicide:
        r -= 300
    if not player_alive and player_suicide:
        r -= 500
    if player_killed_enemy:
        r += 500
    if sectors_cleared_by_player is not None:
        if sectors_cleared_by_player == 0:
            r -= 10
        else:
            r += sectors_cleared_by_player * 50
    return r


if __name__ == "__main__":
    model = QModel()
    epsilon = 0.1
    de = 0.01
    discount = 0.98
    lr = 0.01
    gamma = 0.99
    n_past = 50
    epochs = 10
    episodes = 100
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

    grid_path = Path('.') / 'maps' / 'standard' / 'S.csv'
    grid = np.genfromtxt(grid_path, delimiter=',')
    en1_alg = Algorithm.NONE
    en2_alg = Algorithm.NONE
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
                        path_to_save='qtables/tests/test1/qtable.csv',
                        log_file='qtables/tests/test1/log.csv')
