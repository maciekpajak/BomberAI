import time
from typing import Callable, Tuple

import pygame
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..game import Algorithm
from ..game.game import Game
from ..game.enums.action import Action


class QModel:
    def __init__(self):
        self.qtable = {}

    def load(self, path: str) -> None:
        qtable = pd.read_csv(path, index_col=0)
        index_name = qtable.index.name
        self.qtable = qtable.transpose().to_dict(orient='list')
        self.state_type, self.state_range, self.min_enemy_dist = index_name.split(sep='_')
        self.state_range = int(self.state_range)
        self.min_enemy_dist = int(self.min_enemy_dist)

    def compile(self,
                get_reward: Callable[[bool, Action, bool, bool, int, int], float],
                state_type: str,
                state_range: int,
                min_enemy_dist: int,
                learning_rate=0.1,
                discount=0.98,
                epsilon=0.1,
                de=0.01,
                gamma=0.9,
                n_past_states=10) -> None:
        self.get_reward = get_reward
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.gamma = gamma
        self.de = de
        self.n_past_states = n_past_states
        self.state_type = state_type
        self.state_range = state_range
        self.min_enemy_dist = min_enemy_dist

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
            path_to_save: str = 'qtable.csv',
            log_file: str = 'log.csv') -> pd.DataFrame:
        cols = ['epoch', 'epsilon', 'states_viewed', 'avg_sum_of_rewards', 'win_rate']
        history = pd.DataFrame(columns=cols)
        history.to_csv(log_file, mode='a', index=False, header=cols)

        epsilon = self.epsilon

        for epoch in range(start_epoch, epochs + start_epoch):
            epoch_rewards = []
            for _ in tqdm(range(episodes), desc=f'Epoch {epoch + 1}/{epochs + start_epoch}', unit='game'):
                _, sum_reward = self.play_game(epsilon=epsilon, train=True, show=False)
                epoch_rewards.append(sum_reward)

            if show_game:
                self.play_game(epsilon=epsilon, train=False, show=True)

            win_rate = self.win_rate(100)

            mean_reward = np.mean(epoch_rewards)

            print(f'epsilon: {epsilon:5.5f} - viewed states:{len(self.qtable):5} - avg_sum_of_rewards:{mean_reward:5.1f} - win_rate:{win_rate:2.2f}')

            self.save(path_to_save)
            history = pd.DataFrame([[epoch, epsilon, len(self.qtable), mean_reward, win_rate]], columns=cols)
            history.to_csv(log_file, mode='a', index=False, header=False)

            epsilon = max(0.0, epsilon - self.de)

        return history

    def play_game(self,
                  epsilon: float,
                  train: bool,
                  show: bool) -> Tuple[bool, float]:
        rewards = []
        past_states = []

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
        while not game_over:
            if show:
                dt = clock.tick(15 * speed)
                pygame.event.get()
            else:
                dt = 1000 / (15 * speed)

            state = game.get_state(agent=game.player,
                                   state_type=self.state_type,
                                   state_range=self.state_range,
                                   min_enemy_dist=self.min_enemy_dist)
            if train:
                if state not in self.qtable:
                    self.qtable[state] = np.round(np.random.uniform(0, 0, 6), 3)

                if np.random.random() < epsilon:
                    action = np.random.choice(list(Action), 1, p=[0.17, 0.17, 0.17, 0.17, 0.15, 0.17])[0].value
                else:
                    action = np.argmax(self.qtable[state])

                past_states.insert(0, (state, action))
            else:
                if state not in self.qtable:
                    action = np.random.choice(list(Action), 1, p=[0.17, 0.17, 0.17, 0.17, 0.15, 0.17])[0].value
                else:
                    action = np.argmax(self.qtable[state])

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
                state = game.get_state(agent=enemy,
                                       state_type=enemy.state_type,
                                       state_range=enemy.state_range,
                                       min_enemy_dist=enemy.min_enemy_dist) if enemy.algorithm == Algorithm.Q else None
                enemy.choose_move(grid=game.grid,
                                  bombs=game.bombs,
                                  explosions=game.explosions,
                                  agents=game.agents_on_board,
                                  power_ups=game.power_ups,
                                  state=state)

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
                    future_state = game.get_state(agent=game.player,
                                                  state_type=self.state_type,
                                                  state_range=self.state_range,
                                                  min_enemy_dist=self.min_enemy_dist)
                    if future_state not in self.qtable:
                        self.qtable[future_state] = np.round(np.random.uniform(0, 0, 6), 3)
                    future_action_value = max(self.qtable[future_state])

                    for i, (state, action) in enumerate(past_states):
                        new_value = self.qtable[state][action] + self.learning_rate * (
                                self.gamma ** i * reward + (self.discount * future_action_value) - self.qtable[state][
                            action])
                        self.qtable[state][action] = round(new_value, 3)

                else:
                    for i, (state, action) in enumerate(past_states):
                        new_value = self.qtable[state][action] + self.learning_rate * (
                                self.gamma ** i * reward - self.qtable[state][action])
                        self.qtable[state][action] = round(new_value, 3)

                rewards.append(reward)
            # ----------------------------------------

            if len(past_states) > self.n_past_states:
                past_states.pop()

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
            win, _ = self.play_game(0.01, False, False)
            if win:
                wins += 1
        return wins / n_games

    def save(self, path: str) -> None:
        df = pd.DataFrame.from_dict(self.qtable, orient='index')
        index_name = f'{self.state_type}_{self.state_range}_{self.min_enemy_dist}'
        df.index.rename(index_name, inplace=True)
        df.to_csv(path)
