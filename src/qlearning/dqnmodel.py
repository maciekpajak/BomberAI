import time
from collections import deque
from typing import Callable, Tuple

import pygame
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import tensorflow as tf
import random
from simple_chalk import green, red, yellow

from ..game import Algorithm
from ..game.game import Game
from ..game.enums.action import Action

class DQNModel:
    def __init__(self, model):
        self.main_model = tf.keras.models.clone_model(model)
        self.main_model = model.__class__.from_config(model.get_config())

        self.target_model = tf.keras.models.clone_model(model)
        self.target_model = model.__class__.from_config(model.get_config())

        self.target_model.set_weights(self.main_model.get_weights())
        self.steps_to_update_target_model = 0
        self.update_target_every = 100
        self.batch_size = 128
        self.log_file = None
        self.past_frames = 4
        self.verbosity = False

    def load(self, path: str) -> None:
        self.main_model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)

    def compile(self,
                loss,
                optimizer,
                metrics,
                get_reward: Callable[[bool, Action, bool, bool, int, int], float],
                discount=0.98,
                learning_rate=0.1) -> None:
        self.get_reward = get_reward
        self.discount = discount
        self.learning_rate=learning_rate

        self.main_model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

        self.target_model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

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
            episodes: int = 1000,
            show_game: bool = False,
            update_target_every: int = 100,
            past_frames:int = 4,
            batch_size: int = 4,
            train_clip = (-1,1),
            save_every = 50,
            show_every = 50,
            path_to_save: str = 'model.h5',
            history_log: str= 'history.csv',
            log_file: str = 'log.log',
            verbosity=False,
            ) -> pd.DataFrame:

        self.verbosity = verbosity

        cols = ['epoch', 'epsilon', 'mean_reward', 'mean_loss', 'mean_acc']
        history = pd.DataFrame(columns=cols)
        history.to_csv(history_log, mode='w', index=False, header=cols)

        self.log_file = log_file
        open(log_file, "w").close()

        self.batch_size = batch_size

        self.steps_to_update_target_model = 0
        self.update_target_every = update_target_every

        self.past_frames = past_frames

        self.train_clip = train_clip

        epsilon = 1
        max_epsilon = 1  # You can't explore more than 100% of the time
        min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
        decay = 0.01

        self.replay_memory = deque(maxlen=50_000)
        for e in range(episodes):
            _, mean_reward, mean_loss, mean_acc = self.play_game(epsilon=epsilon, train=True, show=False)
            if show_game and (e + 1) % save_every == 0:
                self.save(path_to_save)
            if show_game and (e + 1) % show_every == 0:
                self.play_game(epsilon=epsilon, train=False, show=True)
            if self.verbosity:
                print(f'Episode {e + 1}/{episodes} - epsilon: {epsilon:5.5f} - loss: {mean_loss:5.5f} - mean acc: {mean_acc:5.5f} - mean reward:{mean_reward:5.1f}\n')
                print('\n' + '=' * 50)
            with open(log_file, "a") as f:
                f.write(f'Episode {e + 1}/{episodes} - epsilon: {epsilon:5.5f} - loss: {mean_loss:5.5f} - mean acc: {mean_acc:5.5f} - mean reward:{mean_reward:5.1f}\n')
                f.write('\n' + '=' * 50 + '\n')

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * e)

            history = pd.DataFrame([[e, epsilon, mean_reward, mean_loss, mean_acc]], columns=cols)
            history.to_csv(history_log, mode='a', index=False, header=False)

        return history

    def train_model(self, replay_memory):
        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return 0, 0

        mini_batch = random.sample(replay_memory, self.batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.main_model.predict(tf.convert_to_tensor(current_states, dtype=tf.float32), verbose=0)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(tf.convert_to_tensor(new_current_states, dtype=tf.float32), verbose=0)

        X = []
        Y = []
        for index, (state, action, reward, future_state, alive) in enumerate(mini_batch):
            if self.train_clip:
                reward = np.clip(reward, self.train_clip[0], self.train_clip[1])

            if alive:
                max_future_q = reward + self.discount * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q

            X.append(state)
            Y.append(current_qs)
        history = self.main_model.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)
        mean_acc = np.mean(history.history['accuracy'])
        mean_loss = np.mean(history.history['loss'])
        return mean_loss, mean_acc

    def play_game(self,
                  epsilon: float,
                  train: bool,
                  show: bool,) -> Tuple[bool, float, float, float]:
        rewards = []
        loss_list = []
        accuracy_list = []
        observations = deque(maxlen=self.past_frames)
        for _ in range(4):
            observations.append(np.zeros((500,500,3)))
        action_list = []

        pygame.init()
        if show:
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
            max_playing_time = 100
            surface = pygame.Surface(size=(500, 500))

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

        game.init_sprites()
        game.draw(surface)

        state = pygame.surfarray.array3d(surface)
        observations.append(state)

        game_over = False

        start_time = time.time()
        while not game_over:
            self.steps_to_update_target_model += 1
            if show:
                dt = clock.tick(15 * speed)
                pygame.event.get()
            else:
                dt = 1000 / (15 * speed)

            history_state = np.concatenate([o for o in observations], axis=2)
            if train:
                if np.random.random() < epsilon:
                    action = np.random.choice(list(Action), 1, p=[0.17, 0.17, 0.17, 0.17, 0.15, 0.17])[0].value
                else:
                    preds = self.target_model.predict(np.expand_dims(history_state, axis=0), verbose=0)
                    action = np.argmax(preds)
            else:
                action = np.argmax(self.target_model.predict(np.expand_dims(history_state, axis=0), verbose=0))

            action_list.append(Action(action))
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
                                           min_enemy_dist=enemy.min_enemy_dist) if enemy.algorithm == Algorithm.Q else None
                enemy.choose_move(grid=game.grid,
                                  bombs=game.bombs,
                                  explosions=game.explosions,
                                  agents=game.agents_on_board,
                                  power_ups=game.power_ups,
                                  state=tmp_state,
                                  surface=surface)

            # update bombs
            suicide, kills, destroyed_boxes = game.update_bombs(dt)
            game.draw(surface)
            observations.append(pygame.surfarray.array3d(surface))

            before_training = time.time()
            # ----------------------------------------
            reward = self.get_reward(game.player.alive,
                                     Action(action),
                                     is_move_possible,
                                     suicide,
                                     kills,
                                     destroyed_boxes)
            rewards.append(reward)
            if train:
                # reward = np.sum([0.98**(len(tmp_rewards) - i) * r for i, r in enumerate(tmp_rewards)])
                fhistory_state = np.concatenate([o for o in observations], axis=2)
                self.replay_memory.append([history_state, action, reward, fhistory_state, game.player.alive])

                if self.steps_to_update_target_model % 4 or not game.player.alive:
                    loss, acc = self.train_model(self.replay_memory)
                    loss_list.append(loss)
                    accuracy_list.append(acc)

                if self.steps_to_update_target_model >= 100 and game.player.alive:
                    # print('Copying main network weights to the target network weights')
                    self.target_model.set_weights(self.main_model.get_weights())
                    self.steps_to_update_target_model = 0

            after_training = time.time()

            # ----------------------------------------
            if show:
                pygame.display.update()

            if not game_over:
                game.playing_time = time.time() - start_time - (after_training - before_training)
                game_over = game.check_end_game()

            with open(self.log_file, 'a') as f:
                dead_type = ''
                if not game.player.alive:
                    if suicide:
                        dead_type = 'suicide'
                    else:
                        dead_type = 'killed by enemy'
                move = green(str(Action(action)).rjust(20)) if is_move_possible else red(str(Action(action)).rjust(20))
                if self.verbosity:
                    print(f'{game.playing_time:3.3f} | {move:20} | {reward:3.3f} {rewards[-1]:3.3f} | '
                          f'{destroyed_boxes:3} - {kills:3} | '
                          f'{red("dead") if not game.player.alive else green("alive"):>5} {dead_type:>10}')
                f.write(f'{game.playing_time:3.3f} | {str(Action(action)).rjust(20) if is_move_possible else str(Action(action)).rjust(20):20} | {reward:3.3f} {rewards[-1]:3.3f} | '
                      f'{destroyed_boxes:3} - {kills:3} | '
                      f'{red("dead") if not game.player.alive else green("alive"):>5} {dead_type:>10}\n')
        if show:
            pygame.display.quit()
            pygame.quit()

        player_win = True if game.player.alive and game.playing_time <= game.max_playing_time else False

        mean_loss = 0 if loss_list is [] else float(np.mean(loss_list))
        mean_acc = 0 if accuracy_list is [] else float(np.mean(accuracy_list))
        mean_reward = 0 if rewards is [] else float(np.mean(rewards))

        return player_win, mean_reward, mean_loss, mean_acc,

    def win_rate(self, n_games: int = 100) -> float:
        wins = 0
        for _ in tqdm(range(n_games), desc=f'Test on {n_games} games', unit='game'):
            win, _, _, _ = self.play_game(0.01, False, False, )
            if win:
                wins += 1
        return wins / n_games

    def save(self, path: str) -> None:
        self.target_model.save(path)
