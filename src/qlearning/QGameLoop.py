import sys
import time
from enum import Enum

import pygame
import numpy as np
import pandas as pd

from ..game import Algorithm
from ..game.game import Game, GRID_BASE
from ..game.enums.action import Action


class GameLoop:
    clock = None
    player_alg = Algorithm.PLAYER
    en1_alg = Algorithm.DFS
    en2_alg = Algorithm.NONE
    en3_alg = Algorithm.NONE
    show_path = True
    surface = None
    TILE_SIZE = 4

    def __init__(self, speed, show_game=False, surface=None, tile_size=4):
        self.GAMEOVER = False
        self.show_game = show_game
        self.last_reward = None
        self.TILE_SIZE = tile_size
        if show_game:
            self.surface = surface

        self.g = Game(self.show_path, self.player_alg, self.en1_alg, self.en2_alg, self.en3_alg, self.TILE_SIZE, speed)
        self.speed = speed
        self.g.init_sprites()

    def get_state(self):
        # return self.g.get_state()
        # return self.g.get_9grid_state()
        return self.g.get_9crossgrid_state()
        # return self.g.get_5grid_state()

    def get_reward(self, action, player_killed_enemy, sectors_cleared_by_player, life_time, is_move_possible):
        r = -0.1
        if self.g.player.bomb_limit == 0 and action == Action.PLANT_BOMB:
            r -= 5
        if not is_move_possible:
            r -= 3
        if not self.g.player.life:
            r -= 300
        if player_killed_enemy:
            r += 500
        if sectors_cleared_by_player is not None:
            if sectors_cleared_by_player == 0:
                r -= 10
            else:
                r += sectors_cleared_by_player * 100
        # r += life_time
        return r

    def run(self, qtable, learning_rate=0.5, discount=0.98, epsilon=0.1):
        self.g.generate_map(GRID_BASE)
        clock = pygame.time.Clock()
        life_time = 1
        rewards = []
        while not self.GAMEOVER:
            # life_time += 0.01
            if not self.g.player.life:
                break
            else:
                dt = clock.tick(int(15 * self.speed))
                for en in self.g.enemy_list:
                    en.make_move(self.g.grid, self.g.bombs, self.g.explosions, self.g.ene_blocks)

                state = self.get_state()
                if state not in qtable:
                    qtable[state] = np.random.uniform(0, 1, 6)

                action = np.argmax(qtable[state])

                if np.random.random() < epsilon:
                    action = np.random.choice(list(Action), 1, p=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05])[0].value

                is_move_possible = self.g.move_player(Action(action))
                player_killed_enemy, sectors_cleared_by_player = self.g.update_bombs(dt)
                reward = self.get_reward(action, player_killed_enemy, sectors_cleared_by_player, life_time, is_move_possible)
                if self.show_game:
                    self.g.draw(self.surface)

                if self.g.player.life:
                    future_state = self.get_state()
                    if future_state not in qtable:
                        qtable[future_state] = np.random.uniform(0, 1, 6)
                    future_action_value = max(qtable[future_state])

                    qtable[state][action] = qtable[state][action] + learning_rate * (
                            reward + (discount * future_action_value) - qtable[state][action])

                else:
                    qtable[state][action] = qtable[state][action] - 300
                    self.GAMEOVER = True

                rewards.append(reward)
                if np.sum([enemy.life for enemy in self.g.enemy_list]) == 0:
                    self.GAMEOVER = True

        return np.mean(rewards)