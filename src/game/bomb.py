from src.game.enums.tile import Tile


class Bomb:
    frame = 0
    speed = 1

    def __init__(self, range, x, y, map, bomber, speed):
        self.range = range
        self.pos_x = x
        self.pos_y = y
        self.speed = speed
        self.time_to_explode = 1500 / speed
        self.bomber = bomber
        self.sectors = []
        self.get_range(map)

    def update(self, dt):

        self.time_to_explode = self.time_to_explode - dt

        if self.time_to_explode < (500 / self.speed):
            self.frame = 2
        elif self.time_to_explode < (1000 / self.speed):
            self.frame = 1

    def get_range(self, grid):

        self.sectors.append([self.pos_x, self.pos_y])

        for x in range(1, self.range):
            if grid[self.pos_x + x][self.pos_y] == Tile.SOLID:
                break
            elif grid[self.pos_x + x][self.pos_y] == Tile.GROUND or grid[self.pos_x + x][self.pos_y] == Tile.POWER_UP:
                self.sectors.append([self.pos_x + x, self.pos_y])
            elif grid[self.pos_x + x][self.pos_y] == Tile.BOX:
                self.sectors.append([self.pos_x + x, self.pos_y])
                break
        for x in range(1, self.range):
            if grid[self.pos_x - x][self.pos_y] == Tile.SOLID:
                break
            elif grid[self.pos_x - x][self.pos_y] == Tile.GROUND or grid[self.pos_x - x][self.pos_y] == Tile.POWER_UP:
                self.sectors.append([self.pos_x - x, self.pos_y])
            elif grid[self.pos_x - x][self.pos_y] == Tile.BOX:
                self.sectors.append([self.pos_x - x, self.pos_y])
                break
        for x in range(1, self.range):
            if grid[self.pos_x][self.pos_y + x] == Tile.SOLID:
                break
            elif grid[self.pos_x][self.pos_y + x] == Tile.GROUND or grid[self.pos_x][self.pos_y + x] == Tile.POWER_UP:
                self.sectors.append([self.pos_x, self.pos_y + x])
            elif grid[self.pos_x][self.pos_y + x] == Tile.BOX:
                self.sectors.append([self.pos_x, self.pos_y + x])
                break
        for x in range(1, self.range):
            if grid[self.pos_x][self.pos_y - x] == Tile.SOLID:
                break
            elif grid[self.pos_x][self.pos_y - x] == Tile.GROUND or grid[self.pos_x][self.pos_y - x] == Tile.POWER_UP:
                self.sectors.append([self.pos_x, self.pos_y - x])
            elif grid[self.pos_x][self.pos_y - x] == Tile.BOX:
                self.sectors.append([self.pos_x, self.pos_y - x])
                break
