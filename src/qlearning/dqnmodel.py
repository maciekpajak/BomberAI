import time
from collections import deque
from typing import Callable, Tuple

import pygame
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import random

from ..game import Algorithm
from ..game.game import Game
from ..game.enums.action import Action

def create_model(input_shape):
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape, kernel_initializer=init),
        # tf.keras.layers.Dense(16, activation='relu', kernel_initializer=init),
        tf.keras.layers.Dense(6, activation='linear', kernel_initializer=init)
    ])
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

class DQNModel:
    def __init__(self):
        self.main_model: tf.keras.models.Model = None
        self.target_model: tf.keras.models.Model = None
        self.steps_to_update_target_model  = 0
        self.update_every = 100

    def load(self, path: str) -> None:
        self.main_model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)

    def compile(self,
                get_reward: Callable[[bool, Action, bool, bool, int, int], float],
                state_type: str,
                state_range: int,
                min_enemy_dist: int,
                discount=0.98,
                epsilon=0.1,
                de=0.01,) -> None:

        self.get_reward = get_reward
        self.state_type = state_type
        self.state_range = state_range
        self.min_enemy_dist = min_enemy_dist
        self.epsilon = epsilon
        self.de = de
        self.discount = discount

    def set_game(self, grid: np.ndarray,
                 en1_alg: Algorithm,
                 en2_alg: Algorithm,
                 en3_alg: Algorithm,
                 box_density: int | Tuple[int, int] = 5,
                 shuffle_positions: bool = True,
                 max_playing_time=120) -> None:
        self.grid = grid
        self.en1_alg = en1_alg
        self.en2_alg = en2_alg
        self.en3_alg = en3_alg
        self.box_density = box_density
        self.shuffle_positions = shuffle_positions
        self.max_playing_time = max_playing_time

    def fit(self,
            epochs: int = 10,
            episodes: int = 1000,
            start_epoch: int = 0,
            show_game: bool = False,
            update_every: int=100,
            batch_size: int=4,
            path_to_save: str = 'model.h5',
            log_file: str = 'log.csv') -> pd.DataFrame:

        cols = ['epoch', 'epsilon', 'avg_sum_of_rewards', 'win_rate']
        history = pd.DataFrame(columns=cols)
        history.to_csv(log_file, mode='a', index=False, header=cols)

        epsilon = self.epsilon

        self.main_model = create_model(input_shape=(242,))
        self.target_model = create_model(input_shape=(242,))
        self.target_model.set_weights(self.main_model.get_weights())

        self.steps_to_update_target_model = 0
        self.update_every = update_every
        for epoch in range(start_epoch, epochs + start_epoch):
            epoch_rewards = []
            for _ in tqdm(range(episodes), desc=f'Epoch {epoch + 1}/{epochs + start_epoch}', unit='game'):
                _, sum_reward = self.play_game(epsilon=epsilon, train=True, batch_size=batch_size, show=False)
                epoch_rewards.append(sum_reward)


            self.save(path_to_save)

            if show_game:
                self.play_game(epsilon=epsilon, train=False, batch_size=batch_size, show=True)

            win_rate = self.win_rate(1)

            mean_reward = np.mean(epoch_rewards)

            print(f'epsilon: {epsilon:5.5f} - loss: ??? - avg_sum_of_rewards:{mean_reward:5.1f} - win_rate:{win_rate:2.2f}')

            history = pd.DataFrame([[epoch, epsilon, mean_reward, win_rate]], columns=cols)
            history.to_csv(log_file, mode='a', index=False, header=False)

            epsilon = max(0.0, epsilon - self.de)
            # epsilon = epsilon * np.exp(-self.de * epoch)

        return history


    def train_model(self, replay_memory):
        learning_rate = 0.7  # Learning rate
        discount_factor = 0.618

        # MIN_REPLAY_SIZE = 32
        # if len(replay_memory) < MIN_REPLAY_SIZE:
        #     return

        batch_size = 32
        mini_batch = random.sample(replay_memory, min(len(replay_memory), batch_size))
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.main_model.predict(current_states, verbose=0)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        Y = []
        for index, (state, action, reward, future_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(state)
            Y.append(current_qs)
        self.main_model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=1, shuffle=True)
        return

    def play_game(self,
                  epsilon: float,
                  train: bool,
                  batch_size: int,
                  show: bool) -> Tuple[bool, float]:
        rewards = []
        if show:
            pygame.init()
            pygame.display.init()
            clock = pygame.time.Clock()
            SCALE = 500 / len(self.grid)
            surface = pygame.display.set_mode((500, 500))
            speed = 1
            max_playing_time = self.max_playing_time
        else:
            clock = pygame.time.Clock()
            SCALE = 1
            surface = None
            speed = 1
            max_playing_time = 10

        game = Game(grid=self.grid,
                    player_alg=Algorithm.PLAYER,
                    en1_alg=self.en1_alg,
                    en2_alg=self.en2_alg,
                    en3_alg=self.en3_alg,
                    scale=SCALE,
                    speed=speed,
                    show_path=False,
                    box_density=self.box_density,
                    shuffle_positions=self.shuffle_positions,
                    max_playing_time=max_playing_time)

        if show:
            game.init_sprites()

        game_over = False
        start_time = time.time()
        replay_memory=deque(maxlen=50_000)
        while not game_over:
            self.steps_to_update_target_model += 1
            if show:
                dt = clock.tick(15 * speed)
                pygame.event.get()
            else:
                dt = 1000 / (15 * speed)

            state = game.get_full_state_as_list(agent=game.player)
            if train:
                if np.random.random() < epsilon:
                    action = np.random.choice(list(Action), 1, p=[0.17, 0.17, 0.17, 0.17, 0.15, 0.17])[0].value
                else:
                    preds = self.target_model.predict(np.expand_dims(state,axis=0), verbose=0)
                    action = np.argmax(preds)
            else:
                action = np.argmax(self.target_model.predict(np.expand_dims(state,axis=0),verbose=0))

            # move player
            is_move_possible = game.player.move(action=Action(action),
                                                grid=game.grid,
                                                bombs=game.bombs,
                                                enemies=game.enemy_list,
                                                power_ups=game.power_ups)

            # move enemies
            for enemy in game.enemy_list:
                if not enemy.alive:
                    continue
                tmp_state = game.get_state(agent=enemy,
                                       state_type=enemy.state_type,
                                       state_range=enemy.state_range,
                                       min_enemy_dist=enemy.min_enemy_dist,
                                       for_nn=False) if enemy.algorithm == Algorithm.Q else None
                enemy.choose_move(grid=game.grid,
                                  bombs=game.bombs,
                                  explosions=game.explosions,
                                  agents=game.agents_on_board,
                                  power_ups=game.power_ups,
                                  state=tmp_state)

            # update bombs
            suicide, kills, destroyed_boxes = game.update_bombs(dt)

            # ----------------------------------------
            if train:  # update qtable
                reward = self.get_reward(game.player.alive,
                                         Action(action),
                                         is_move_possible,
                                         suicide,
                                         kills,
                                         destroyed_boxes)

                if game.player.alive:
                    future_state = game.get_full_state_as_list(agent=game.player)

                    replay_memory.append([state, action, reward, future_state, game.player.alive])
                else:
                    replay_memory.append([state, action, reward, state, game.player.alive])

                if self.steps_to_update_target_model % batch_size or not game.player.alive:
                    self.train_model(replay_memory)

                if self.steps_to_update_target_model >= self.update_every and game.player.alive:
                    print('Copying main network weights to the target network weights')
                    self.target_model.set_weights(self.main_model.get_weights())
                    self.steps_to_update_target_model = 0

                rewards.append(reward)
            # ----------------------------------------
            if show:
                game.draw(surface)

            if not game_over:
                game.playing_time = time.time() - start_time
                game_over = game.check_end_game()

        if show:
            pygame.display.quit()
            pygame.quit()

        player_win = True if game.player.alive and game.playing_time <= game.max_playing_time else False
        return player_win, float(np.sum(rewards))

    def win_rate(self, n_games: int = 100) -> float:
        wins = 0
        for _ in tqdm(range(n_games), desc=f'Test on {n_games} games', unit='game'):
            win, _ = self.play_game(0.01, False, 4, False,)
            if win:
                wins += 1
        return wins / n_games

    def save(self, path: str) -> None:
        self.target_model.save(path)
