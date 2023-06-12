import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from tqdm import tqdm

from src.game import Algorithm, Action
from src.qlearning.dqnmodel import DQNModel


def get_reward(player_alive, action, is_move_possible, suicide, kills, destroyed_boxes):
    r = -1/23
    if action == Action.NO_ACTION:
        r += -1/23
    if not player_alive and not suicide:
        r -= 100
    if not player_alive and suicide:
        r -= 500
    r += kills * 300
    r += destroyed_boxes * 50
    return r


if __name__ == "__main__":
    model = DQNModel()
    discount = 0.98
    episodes = 1000
    model.compile(get_reward=get_reward,
                  discount=discount)

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

    history = model.fit(episodes=episodes,
                        show_game=True,
                        update_target_every=1000,
                        batch_size=128,
                        path_to_save='qtables/dqnmodels/modelM.h5',
                        log_file='qtables/dqnmodels/log.csv')
