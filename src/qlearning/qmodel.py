import time
from pathlib import Path
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

    def load(self, path):
        qtable = pd.read_csv(path)
        self.qtable = qtable.transpose().to_dict(orient='list')

    def compile(self, get_reward: Callable, learning_rate=0.5,
                discount=0.98,
                epsilon=0.1,
                de=0.001,
                gamma=0.9,
                n_past_states=10,
                state_type='5cross'):
        self.get_reward = get_reward
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.gamma = gamma
        self.de = de
        self.n_past_states = n_past_states
        self.state_type=state_type

    def set_game(self, grid: np.ndarray,
                 en1_alg: Algorithm,
                 en2_alg: Algorithm,
                 en3_alg: Algorithm,
                 training_speed: float = 1000,
                 box_density: int | Tuple[int, int] = 5,
                 shuffle_positions: bool = True,
                 max_playing_time=120,
                 state_type='5cross'):
        self.grid = grid
        self.en1_alg = en1_alg
        self.en2_alg = en2_alg
        self.en3_alg = en3_alg
        self.training_speed = training_speed
        self.box_density = box_density
        self.shuffle_positions = shuffle_positions
        self.max_playing_time = max_playing_time

        self.state_type = state_type

    def fit(self, epochs=10, episodes=1000, start_epoch=0, show_game=False, path_to_save='qtable.csv',
            log_file='log.csv'):
        epsilon = self.epsilon
        cols = ['epoch', 'states_viewed', 'average_sum_of_rewards', 'win_rate']
        history = pd.DataFrame(columns=cols)
        history.to_csv(log_file, mode='w', index=False, header=cols)
        average_sum_of_rewards = []
        states_viewed = []
        for epoch in range(start_epoch, epochs + start_epoch):
            epoch_rewards = []
            for _ in tqdm(range(episodes), desc=f'Epoch {epoch}/{epochs + start_epoch}', unit='game'):
                sum_reward = self.play_game(learning_rate=self.learning_rate,
                                            discount=self.discount,
                                            gamma=self.gamma,
                                            epsilon=epsilon,
                                            n_past_states=self.n_past_states)
                epoch_rewards.append(sum_reward)
            if show_game:
                self.show_game(1)
            print('Playing 100 games...')
            win_rate = self.win_rate(100)
            mean_reward = np.mean(epoch_rewards)
            states_viewed.append(len(self.qtable))
            average_sum_of_rewards.append(mean_reward)
            print(f'e: {epsilon:5.5f} - viewed states:{len(self.qtable):5}  - avg_sum_of_rwds:{mean_reward:5.3f} - wr:{win_rate:2.2f}')
            epsilon = max(0.0, epsilon - self.de)

            self.save(path_to_save)
            history = pd.concat([history,pd.DataFrame([[epoch, len(self.qtable), mean_reward, win_rate]], columns=cols)])
            history.to_csv(log_file, mode='w', index=False, header=False)

        return history

    def play_game(self, learning_rate, discount, gamma, epsilon, n_past_states):
        clock = pygame.time.Clock()
        rewards = []
        past_states = []
        game_over = False

        game = Game(self.grid, Algorithm.PLAYER, self.en1_alg, self.en2_alg, self.en3_alg,
                    1, self.training_speed, False, self.box_density,
                    self.shuffle_positions, self.max_playing_time)

        start_time = time.time()
        while not game_over:
            if not game_over:
                game.playing_time = time.time() - start_time
                game_over = game.check_end_game()
            dt = 1000 / (15 * self.training_speed)

            state = game.get_state(game.player)
            if state not in self.qtable:
                self.qtable[state] = np.round(np.random.uniform(0, 0, 6), 3)

            if np.random.random() < epsilon:
                action = np.random.choice(list(Action), 1, p=[0.2, 0.2, 0.2, 0.2, 0.0, 0.2])[0].value
            else:
                action = np.argmax(self.qtable[state])

            past_states.insert(0, (state, action))

            is_move_possible = game.player.move(Action(action), game.grid, game.bombs, game.enemy_list, game.power_ups)

            for en in game.enemy_list:
                state = game.get_state(en)
                en.choose_move(game.grid, game.bombs, game.explosions, game.agents_on_board,
                               state)

            player_killed_enemy, sectors_cleared_by_player = game.update_bombs(dt)

            reward = self.get_reward(game.player.alive, action, is_move_possible, player_killed_enemy,
                                     sectors_cleared_by_player)

            if game.player.alive:
                future_state = game.get_state(game.player)
                if future_state not in self.qtable:
                    self.qtable[future_state] = np.round(np.random.uniform(0, 0, 6), 3)
                future_action_value = max(self.qtable[future_state])

                for i, (state, action) in enumerate(past_states):
                    new_value = self.qtable[state][action] + learning_rate * (
                            gamma ** i * reward + (discount * future_action_value) - self.qtable[state][action])
                    self.qtable[state][action] = round(new_value, 3)

            else:
                for i, (state, action) in enumerate(past_states):
                    new_value = self.qtable[state][action] + learning_rate * (
                            gamma ** i * reward - self.qtable[state][action])
                    self.qtable[state][action] = round(new_value, 3)

            if len(past_states) > n_past_states:
                past_states.pop()

            rewards.append(reward)

        return np.sum(rewards)

    def show_game(self, speed):
        pygame.init()
        clock = pygame.time.Clock()
        pygame.display.init()
        INFO = pygame.display.Info()
        WINDOW_SIZE = (500, 500)
        SCALE = WINDOW_SIZE[0] / len(self.grid)
        surface = pygame.display.set_mode(WINDOW_SIZE)

        game = Game(self.grid,
                    Algorithm.PLAYER, self.en1_alg, self.en2_alg, self.en3_alg,
                    SCALE, 1, False, self.box_density,
                    self.shuffle_positions, self.max_playing_time)
        game.init_sprites()
        game_over = False
        start_time = time.time()
        while not game_over:
            if not game_over:
                game.playing_time = time.time() - start_time
                game_over = game.check_end_game()
            dt = clock.tick(15 * speed)

            state = game.get_state(game.player)

            if state not in self.qtable:
                action = np.random.choice(list(Action), 1, p=[0.2, 0.2, 0.2, 0.2, 0.0, 0.2])[0]
            else:
                action = Action(np.argmax(self.qtable[state]))

            game.player.move(action, game.grid, game.bombs, game.enemy_list,
                             game.power_ups)

            for en in game.enemy_list:
                state = game.get_state(en)
                en.choose_move(game.grid, game.bombs, game.explosions, game.agents_on_board,
                               state)

            game.update_bombs(dt)

            game.draw(surface)
        pygame.display.quit()

    def win_rate(self, n_games:int=100) -> float:
        wins = 0
        for _ in range(n_games):
            clock = pygame.time.Clock()
            game = Game(self.grid,
                        Algorithm.PLAYER, self.en1_alg, self.en2_alg, self.en3_alg,
                        1, self.training_speed, False, self.box_density,
                        self.shuffle_positions, self.max_playing_time)
            game_over = False
            start_time = time.time()
            while not game_over:
                if not game_over:
                    game.playing_time = time.time() - start_time
                    game_over = game.check_end_game()
                dt = 1000 / (15 * self.training_speed)

                state = game.get_state(game.player)

                if state not in self.qtable:
                    action = np.random.choice(list(Action), 1, p=[0.2, 0.2, 0.2, 0.2, 0.0, 0.2])[0]
                else:
                    action = Action(np.argmax(self.qtable[state]))

                game.player.move(action, game.grid, game.bombs, game.enemy_list,
                                 game.power_ups)

                for en in game.enemy_list:
                    state = game.get_state(en)
                    en.choose_move(game.grid, game.bombs, game.explosions, game.agents_on_board,
                                   state)

                game.update_bombs(dt)

            if game.player.alive and game.playing_time <= game.max_time:
                wins += 1
        return wins / n_games

    def save(self, path):
        qtable_df = pd.DataFrame(self.qtable.values(), index=self.qtable.keys())
        qtable_df.to_csv(path)
