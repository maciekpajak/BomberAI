from random import random

from src.game.enums import Tile
from src.game.enums.power_up_type import PowerUpType
from src.game.power_up import PowerUp


class Explosion:

    bomber = None

    def __init__(self, x, y, r, speed):
        self.sourceX = x
        self.sourceY = y
        self.speed= speed
        self.range = r
        self.time = 300 / (speed ** 0.5)
        self.frame = 0
        self.sectors = []

    def explode(self, map, bombs, b, power_ups):

        self.bomber = b.bomber
        self.sectors.extend(b.sectors)
        bombs.remove(b)
        self.bomb_chain(bombs, map, power_ups)
        return self.bomber

    def bomb_chain(self, bombs, grid, power_ups):

        for s in self.sectors:
            for x in power_ups:
                if x.pos_x == s[0] and x.pos_y == s[1]:
                    power_ups.remove(x)

            for x in bombs:
                if x.pos_x == s[0] and x.pos_y == s[1]:
                    grid[x.pos_x][x.pos_y] = Tile.GROUND
                    x.bomber.bomb_limit += 1
                    self.explode(grid, bombs, x, power_ups)

    def clear_sectors(self, grid, power_ups):

        sectors_cleared = 0
        for i in self.sectors:
            if grid[i[0]][i[1]] == Tile.BOX:
                sectors_cleared += 1
                # uncomment to enable powerups
                #
                # r = random.randint(0, 9)
                # if r == 0:
                #     power_ups.append(PowerUp(i[0], i[1], PowerUpType.BOMB))
                # elif r == 1:
                #     power_ups.append(PowerUp(i[0], i[1], PowerUpType.FIRE))

            grid[i[0]][i[1]] = Tile.GROUND
        return sectors_cleared

    def update(self, dt):

        self.time = self.time - dt

        if self.time < (100 / self.speed):
            self.frame = 2
        elif self.time < (200 / self.speed):
            self.frame = 1
