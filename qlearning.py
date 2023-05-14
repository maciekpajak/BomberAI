import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from tqdm import tqdm

from src.game import Algorithm, Action
from src.qlearning.qmodel import QModel


def get_reward(player_alive, action, is_move_possible, player_killed_enemy, sectors_cleared_by_player):
    r = 0
    if action in list(Action):
        r -= 1
    if not is_move_possible:
        r -= 5
    if not player_alive:
        r -= 300
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
    gamma = 0.9
    n_past = 10
    epochs = 10
    episodes = 10
    training_speed = 1000
    model.compile(get_reward=get_reward,
                  learning_rate=lr,
                  discount=discount,
                  epsilon=epsilon,
                  de=de,
                  gamma=gamma,
                  n_past_states=n_past)

    grid_path = Path('.') / 'maps' / 'standard' / 'XS.csv'
    grid = np.genfromtxt(grid_path, delimiter=',')
    en1_alg = Algorithm.RANDOM
    en2_alg = Algorithm.RANDOM
    en3_alg = Algorithm.RANDOM
    model.set_game(grid=grid,
                   en1_alg=en1_alg, en2_alg=en2_alg, en3_alg=en3_alg,
                   training_speed=training_speed,
                   box_density=(5, 7),
                   shuffle_positions=True)

    history = model.fit(epochs=epochs,
                        episodes=episodes,
                        start_epoch=0,
                        show_game=True,
                        path_to_save='qtables/tests/test2/qtable.csv',
                        log_file='qtables/tests/test2/log.csv')
