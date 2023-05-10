from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
import random

from src.game.enums import Tile
from src.game.enums.action import Action
from src.game.agent import Agent
from src.game.bomb import Bomb
from src.game.node import Node
from src.game.enums.algorithm import Algorithm


class TileType(Enum):
    SAFE = 0
    UNSAFE = 1
    DESTROYABLE = 2
    UNREACHABLE = 3


class Enemy(Agent):
    dire = [[1, 0, Action.RIGHT], [0, 1, Action.DOWN], [-1, 0, Action.LEFT], [0, -1, Action.UP]]

    def __init__(self, x, y, alg, speed):
        super().__init__(x, y, speed)
        self.path = []
        self.movement_path = []
        self.algorithm = alg
        if self.algorithm == Algorithm.Q:
            qtable_path = (Path('.') / 'src' / 'qtable' / 'qtable.csv').resolve()
            self.qtable = pd.read_csv(qtable_path, index_col='State',)
            self.qtable = self.qtable.transpose().to_dict(orient='list')

    def choose_move(self, grid, bombs, explosions, enemies, state):

        if not self.alive:
            return
        if not self.movement_path:
            if self.algorithm == Algorithm.DFS:
                self.dfs(self.create_grid(grid, bombs, explosions, enemies))
            elif self.algorithm == Algorithm.DIJKSTRA:
                raise NotImplementedError
            elif self.algorithm == Algorithm.Q:
                self.q_path(state)
            else:
                self.random_path(self.create_grid(grid, bombs, explosions, enemies))

        action = self.movement_path[0]
        self.move(action, grid, bombs, enemies, None)
        self.movement_path.pop(0)
        self.path.pop(0)

    def random_path(self, grid):
        n = np.random.randint(5, 10)
        path = [[self.pos_x, self.pos_y]]

        for i in range(n):
            random.shuffle(self.dire)
            x_last, y_last = path[-1]
            if grid[x_last][y_last] == TileType.SAFE and self.bomb_limit == 0:  # path to safe place after planting bomb
                self.movement_path.append(Action.NO_ACTION)
                break

            grid[x_last][y_last] = TileType.UNREACHABLE

            good_move_found = False
            if not good_move_found:
                for action_arr in self.dire:
                    if grid[x_last + action_arr[0]][y_last + action_arr[1]] == TileType.SAFE:
                        path.append([x_last + action_arr[0], y_last + action_arr[1]])
                        self.movement_path.append(action_arr[2])
                        good_move_found = True
                        break
            if not good_move_found:
                for action_arr in self.dire:
                    if grid[x_last + action_arr[0]][y_last + action_arr[1]] == TileType.UNSAFE:
                        path.append([x_last + action_arr[0], y_last + action_arr[1]])
                        self.movement_path.append(action_arr[2])
                        good_move_found = True
                        break
            if not good_move_found:
                path.append([x_last, y_last])
                self.movement_path.append(Action.NO_ACTION)
                break

        if self.bomb_limit != 0:
            self.movement_path.append(Action.PLANT_BOMB)
        self.path = path

    def dfs(self, grid):

        new_path = [[self.pos_x, self.pos_y]]
        depth = 0
        self.dfs_rec(grid, new_path, depth)
        self.path = new_path

    def dfs_rec(self, grid, path, depth):

        x_last, y_last = path[-1]
        if depth > 200:
            self.movement_path.append(Action.NO_ACTION)
            return
        # if bomb planted and current tile is safe
        if grid[x_last][y_last] == TileType.SAFE and self.bomb_limit == 0:  # path to safe place after planting bomb
            self.movement_path.append(Action.NO_ACTION)
            return
        # if any destroyable object around
        if grid[x_last + 1][y_last + 0] == TileType.DESTROYABLE or grid[x_last + 0][
            y_last + 1] == TileType.DESTROYABLE or grid[x_last - 1][y_last + 0] == TileType.DESTROYABLE or \
                grid[x_last + 0][y_last - 1] == TileType.DESTROYABLE:
            if self.bomb_limit != 0:
                self.movement_path.append(Action.PLANT_BOMB)
                return

        grid[x_last][y_last] = TileType.UNREACHABLE  # prevent return

        random.shuffle(self.dire)

        good_move_found = False
        if not good_move_found:
            for action_arr in self.dire:
                if grid[x_last + action_arr[0]][y_last + action_arr[1]] == TileType.SAFE:
                    path.append([x_last + action_arr[0], y_last + action_arr[1]])
                    self.movement_path.append(action_arr[2])
                    good_move_found = True
                    break

        if not good_move_found:
            for action_arr in self.dire:
                if grid[x_last + action_arr[0]][y_last + action_arr[1]] == TileType.UNSAFE:
                    path.append([x_last + action_arr[0], y_last + action_arr[1]])
                    self.movement_path.append(action_arr[2])
                    good_move_found = True
                    break

        if not good_move_found:
            if len(self.movement_path) > 0:
                path.pop(0)
                self.movement_path.pop(0)

        self.dfs_rec(grid, path, depth + 1)

    def q_path(self, state):
        if state not in self.qtable:
            action = np.random.choice(list(Action), 1, p=[0.23, 0.23, 0.23, 0.23, 0.0, 0.08])[0]
            print("[Q-Bot] I've never been here!")
        else:
            action = Action(np.argmax(self.qtable[state]))
        self.movement_path.append(action)
        self.path = [[self.pos_x, self.pos_y]]

    def create_grid(self, grid, bombs, explosions, enemies):
        tmp_grid = np.full_like(grid, fill_value=TileType.SAFE, dtype=TileType)

        for b in bombs:
            b.get_range(grid)
            for x in b.sectors:
                tmp_grid[x[0]][x[1]] = TileType.UNSAFE
            tmp_grid[b.pos_x][b.pos_y] = TileType.UNREACHABLE

        for e in explosions:
            for s in e.sectors:
                tmp_grid[s[0]][s[1]] = TileType.UNREACHABLE

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == Tile.SOLID.value:
                    tmp_grid[i][j] = TileType.UNREACHABLE
                elif grid[i][j] == Tile.BOX.value:
                    tmp_grid[i][j] = TileType.DESTROYABLE

        for x in enemies:
            if x == self:
                continue
            elif not x.alive:
                continue
            else:
                tmp_grid[x.pos_x][x.pos_y] = TileType.DESTROYABLE

        return tmp_grid
