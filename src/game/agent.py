import math

import pygame

from src.game import Explosion
from src.game.enums.action import Action
from src.game.bomb import Bomb
from src.game.enums.power_up_type import PowerUpType
from src.game.enums.tile import Tile


class Agent:
    life = True
    pos_x = None
    pos_y = None
    direction = 0
    frame = 0
    animation = []
    bomb_range = 2
    bomb_limit = 1

    def __init__(self, x: int, y: int, tile_size: int, speed: float):
        self.tile_size = tile_size
        self.pos_x = x * self.tile_size
        self.pos_y = y * self.tile_size
        self.speed = speed

    def move(self, action: Action, grid, enemies, power_ups) -> bool:
        dx, dy = 0, 0
        if action == Action.UP:
            dx, dy = 0, -1
        elif action == Action.DOWN:
            dx, dy = 0, 1
        elif action == Action.LEFT:
            dx, dy = -1, 0
        elif action == Action.RIGHT:
            dx, dy = 1, 0
        elif action == Action.NO_ACTION:
            dx, dy = 0, 0

        # for x in enemies:
        #     if x == self:
        #         continue
        #     elif not x.life:
        #         continue
        #     else:
        #         # continue
        #         map[int(x.pos_x / Player.TILE_SIZE)][int(x.pos_y / Player.TILE_SIZE)] = 2

        # zapobiega poruszaniu się miedzy gridem
        if self.pos_x % self.tile_size != 0 and dx == 0:
            return True
        if self.pos_y % self.tile_size != 0 and dy == 0:
            return True

        if action == Action.UP or action == Action.LEFT:
            grid_y = math.ceil(self.pos_y / self.tile_size)
            grid_x = math.ceil(self.pos_x / self.tile_size)
        else:
            grid_y = math.floor(self.pos_y / self.tile_size)
            grid_x = math.floor(self.pos_x / self.tile_size)
        if grid[grid_x + dx][grid_y + dy] == Tile.SOLID.value or grid[grid_x + dx][grid_y + dy] == Tile.BOX.value:
            return False
        else:
            self.pos_x += dx
            self.pos_y += dy

        return True

    def plant_bomb(self, map):
        b = Bomb(self.bomb_range,
                 round(self.pos_x / self.tile_size),
                 round(self.pos_y / self.tile_size),
                 map, self, self.speed)
        return b

    def check_death(self, explosions: list[Explosion]):
        for explosion in explosions:
            for sector in explosion.sectors:
                if int(self.pos_x / self.tile_size) == sector[0] and int(self.pos_y / self.tile_size) == sector[1]:
                    self.life = False
                    return explosion.bomber

    def consume_power_up(self, power_up, power_ups):
        if power_up.type == PowerUpType.BOMB:
            self.bomb_limit += 1
        elif power_up.type == PowerUpType.FIRE:
            self.bomb_range += 1

        power_ups.remove(power_up)

    def load_animations(self, image_path, scale: int):
        front = []
        back = []
        left = []
        right = []
        resize_width = scale
        resize_height = scale

        f1 = pygame.image.load(image_path + 'f0.png')
        f2 = pygame.image.load(image_path + 'f1.png')
        f3 = pygame.image.load(image_path + 'f2.png')

        f1 = pygame.transform.scale(f1, (resize_width, resize_height))
        f2 = pygame.transform.scale(f2, (resize_width, resize_height))
        f3 = pygame.transform.scale(f3, (resize_width, resize_height))

        front.append(f1)
        front.append(f2)
        front.append(f3)

        r1 = pygame.image.load(image_path + 'r0.png')
        r2 = pygame.image.load(image_path + 'r1.png')
        r3 = pygame.image.load(image_path + 'r2.png')

        r1 = pygame.transform.scale(r1, (resize_width, resize_height))
        r2 = pygame.transform.scale(r2, (resize_width, resize_height))
        r3 = pygame.transform.scale(r3, (resize_width, resize_height))

        right.append(r1)
        right.append(r2)
        right.append(r3)

        b1 = pygame.image.load(image_path + 'b0.png')
        b2 = pygame.image.load(image_path + 'b1.png')
        b3 = pygame.image.load(image_path + 'b2.png')

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        back.append(b1)
        back.append(b2)
        back.append(b3)

        l1 = pygame.image.load(image_path + 'l0.png')
        l2 = pygame.image.load(image_path + 'l1.png')
        l3 = pygame.image.load(image_path + 'l2.png')

        l1 = pygame.transform.scale(l1, (resize_width, resize_height))
        l2 = pygame.transform.scale(l2, (resize_width, resize_height))
        l3 = pygame.transform.scale(l3, (resize_width, resize_height))

        left.append(l1)
        left.append(l2)
        left.append(l3)

        self.animation.append(front)
        self.animation.append(right)
        self.animation.append(back)
        self.animation.append(left)
