import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from tqdm import tqdm
import tensorflow as tf

from src.game import Algorithm, Action
from src.qlearning.dqnmodel import DQNModel



def create_model():
    inputs = tf.keras.layers.Input(shape=(500,500, 12))
    x = tf.keras.layers.Resizing(32, 32)(inputs)
    x = tf.keras.layers.Rescaling(127.5, offset=-1)(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='valid')(x)
    # x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='valid')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform())(x)
    x = tf.keras.layers.Dense(6, activation='linear', kernel_initializer=tf.keras.initializers.HeUniform())(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    # print(model.summary())
    return model


def get_reward(player_alive, action, is_move_possible, suicide, kills, destroyed_boxes):
    r = -1/23
    if action == Action.NO_ACTION:
        r += -1/23
    if not is_move_possible:
        r += -1
    if not player_alive and not suicide:
        r -= 300
    if not player_alive and suicide:
        r -= 500
    r += kills * 300
    r += destroyed_boxes * 50
    return r


if __name__ == "__main__":
    model = DQNModel(create_model())
    learning_rate=0.01
    discount = 0.98
    episodes = 1000
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy'],
        get_reward=get_reward,
        discount=discount,
        learning_rate=0.001)

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
                   max_playing_time=450)

    history = model.fit(episodes=episodes,
                        show_game=True,
                        update_target_every=100,
                        past_frames=4,
                        batch_size=32,
                        train_clip=(-100,100),
                        save_every=2,
                        path_to_save='qtables/dqnmodels/modelM.h5',
                        history_log='qtables/dqnmodels/history.csv',
                        log_file='qtables/dqnmodels/log.log')
