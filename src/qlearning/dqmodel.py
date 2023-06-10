import os
import random
import time
from typing import Callable, Tuple
from collections import deque

import pygame
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn

from ..game import Algorithm
from ..game.game import Game
from ..game.enums.action import Action

class DQNetwork(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int = 256, output_shape: int = 6):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape))

    def forward(self, x):
        return self.net(x)


class DQModel:
    def __init__(self):  ####### verify
        self.qtable = {}

    def load(self, path: str) -> None:  ####### verify
        qtable = pd.read_csv(path, index_col=0)
        index_name = qtable.index.name
        self.qtable = qtable.transpose().to_dict(orient='list')
        self.state_type, self.state_range, self.min_enemy_dist = index_name.split(sep='_')
        self.state_range = int(self.state_range)
        self.min_enemy_dist = int(self.min_enemy_dist)

    @staticmethod
    def state_size(state_type: str, state_range: int):
        if state_type == 'full':
            raise ValueError("State \'full\' not implemented for Deep Q-Network")
        elif state_type == 'cross':
            return 1+4*(state_range-1)
        elif state_type == 'square':
            return state_range**2
        elif state_type == 'circle':
            circle_val = 0
            for i in range(-state_range + 1, state_range):
                for j in range(-state_range + 1, state_range):
                    if abs(i) + abs(j) < state_range:
                        circle_val += 1
            return circle_val
        else:
            raise ValueError(" State must be one of: full, cross, square or circle")

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
                n_past_states=10,
                batch_size = 32,
                buffer_size = 50000,
                min_replay_size = 1000,
                target_update_freq = 1000,
                loss = nn.SmoothL1Loss) -> None:
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
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq  #no of game steps of the episode to update target network
        self.loss = loss

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.online_network = DQNetwork(input_shape=self.state_size(state_type, state_range)).to(self.device)
        self.target_network = DQNetwork(input_shape=self.state_size(state_type, state_range)).to(self.device)
        self.online_network.train()
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(params=self.online_network.parameters(), lr=self.learning_rate)  # to define by user
        self.target_network.load_state_dict(self.online_network.state_dict())

    @staticmethod
    def convert_obs(obs: str) -> list[int]:
        list_obs = list(obs)
        enemy_dist_x = int(list_obs[-6] + list_obs[-5] + list_obs[-4])
        enemy_dist_y = int(list_obs[-3] + list_obs[-2] + list_obs[-1])
        state = [int(i) for i in list_obs[:-6]]
        state.append(enemy_dist_x)
        state.append(enemy_dist_y)
        return state

    def init_replay(self, path: str):
        replay_buffer = deque(maxlen=self.buffer_size)
        qtable = pd.read_csv(path, index_col=0, nrows=self.min_replay_size+1)
        qtable = qtable.transpose().to_dict(orient='list')
        iter_qtable = iter(qtable)
        obs = next(iter_qtable)
        for i in range(len(qtable)):  # for Python >= 3.6
            action = np.argmax(qtable[obs])
            reward = max(qtable[obs])  # check !!
            converted_obs = self.convert_obs(obs)
            new_obs = next(iter_qtable)
            converted_new_obs = self.convert_obs(new_obs)
            set = (converted_obs, action, reward, converted_new_obs)
            replay_buffer.append(set)
            obs = new_obs
        return replay_buffer  # firstly from table -> to do: init 1000 actions from game to make sure dependency

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
            path_to_load: str = 'qtable.csv',  # to delete -> loading of states will be during compilin (saving useless in DQN)
            path_to_log: str = 'log.csv',
            path_to_models: str = 'model.pth') -> pd.DataFrame:
        cols = ['epoch', 'epsilon', 'states_viewed', 'avg_sum_of_rewards', 'win_rate']

        paths = [path_to_load, path_to_log, path_to_models]
        for path in paths:
            if '/' in path:
                if '.' in path:
                    if os.path.isdir(os.getcwd() + '/' + path[:path.rfind('/')]):
                        pass
                    else:
                        if '.' in path:
                            os.makedirs(os.getcwd() + '/' + path[:path.rfind('/')])
                elif os.path.isdir(os.getcwd() + '/' + path):
                    pass
                else:
                    os.makedirs(os.getcwd() + '/' + path)

        replay_buffer = self.init_replay(path_to_load)  #init replay to correct (format like in notebook)

        history = pd.DataFrame(columns=cols)
        history.to_csv(path_to_log, mode='a', index=False, header=cols)

        epsilon = self.epsilon

        for epoch in range(start_epoch, epochs + start_epoch):
            epoch_rewards = []
            for _ in tqdm(range(episodes), desc=f'Epoch {epoch + 1}/{epochs + start_epoch}', unit='game'):
                _, sum_reward = self.play_game(epsilon=epsilon, train=True, show=False, replay_buffer=replay_buffer)
                epoch_rewards.append(sum_reward)

            if epoch % 5 == 0:
                model_file = path_to_models + "/" + f"{self.state_type}_{str(self.state_range)}_model_" + str(epoch) + '.pth'
                torch.save(self.target_network.state_dict(destination=model_file))

            if show_game:
                self.play_game(epsilon=epsilon, train=False, show=True, replay_buffer=replay_buffer)

            win_rate = self.win_rate(100)

            mean_reward = np.mean(epoch_rewards)

            print(f'epsilon: {epsilon:5.5f} - viewed states:{len(self.qtable):5} - avg_sum_of_rewards:{mean_reward:5.1f} - win_rate:{win_rate:2.2f}')

            self.save(path_to_load)
            history = pd.DataFrame([[epoch, epsilon, len(self.qtable), mean_reward, win_rate]], columns=cols)
            history.to_csv(path_to_log, mode='a', index=False, header=False)

            epsilon = max(0.0, epsilon - self.de)

        return history

    def play_game(self,
                  epsilon: float,
                  train: bool,
                  show: bool,
                  replay_buffer) -> Tuple[bool, float]:
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
        step = 1  # steps of the episode (to update target_network)
        while not game_over:
            if show:
                dt = clock.tick(15 * speed)
                pygame.event.get()
            else:
                dt = 1000 / (15 * speed)

            # get state but what with replay buffer (Colab DQN)
            state = game.get_state(agent=game.player,
                                   state_type=self.state_type,
                                   state_range=self.state_range,
                                   min_enemy_dist=self.min_enemy_dist,
                                   if_dqm=True)
            if train:
                # if state not in self.qtable:
                #     self.qtable[state] = np.round(np.random.uniform(0, 0, 6), 3)

                if np.random.random() < epsilon:
                    action = np.random.choice(list(Action), 1, p=[0.17, 0.17, 0.17, 0.17, 0.15, 0.17])[0].value
                else:
                    # action = np.argmax(self.qtable[state])
                    with torch.no_grad:
                        action = torch.argmax(self.online_network(state))

                past_states.insert(0, (state, action))
            else:
                if state not in self.qtable:
                    action = np.random.choice(list(Action), 1, p=[0.17, 0.17, 0.17, 0.17, 0.15, 0.17])[0].value
                else:
                    # action = np.argmax(self.qtable[state])
                    with torch.no_grad():
                        action = torch.argmax(self.target_network(state))

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

            if train:
                batch = random.sample(replay_buffer, self.batch_size)

                obss = np.asarray([i[0] for i in batch], dtype=np.int64)
                acts = np.asarray([i[1] for i in batch], dtype=np.int64)
                rews = np.asarray([i[2] for i in batch], dtype=np.float32)
                nobss = np.asarray([i[3] for i in batch], dtype=np.int64)

                obss_t = torch.as_tensor(obss, dtype=torch.int64).to(self.device)
                acts_t = torch.as_tensor(acts, dtype=torch.int64).unsqueeze(-1).to(self.device)
                rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1).to(self.device)
                new_obss_t = torch.as_tensor(nobss, dtype=torch.int64).to(self.device)

                target_q_values = self.target_network(new_obss_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                ############# if done -> by condition if alive


            if len(past_states) > self.n_past_states:
                past_states.pop()

            if show:
                game.draw(surface)

            if step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.online_network.state_dict())
                print("Target network updated")

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

    def save(self, path: str) -> None:  # to modify
        df = pd.DataFrame.from_dict(self.qtable, orient='index')
        index_name = f'{self.state_type}_{self.state_range}_{self.min_enemy_dist}'
        df.index.rename(index_name, inplace=True)
        df.to_csv(path)