import math

import pygame

from src.game.explosion import Explosion
from src.game.enums.action import Action
from src.game.bomb import Bomb
from src.game.enums.power_up_type import PowerUpType
from src.game.enums.tile import Tile


class Agent:

    def __init__(self, x: int, y: int, speed: float):
        self.alive = True
        self.bomb_range = 2
        self.bomb_limit = 1
        self.pos_x = x
        self.pos_y = y
        self.speed = speed
        self.direction = 2
        self.frame = 0
        self.animation = []

    def choose_move(self, grid, bombs, explosions, agents, state):
        raise NotImplementedError

    def move(self, action: Action, grid, bombs, enemies, power_ups) -> bool:
        dx, dy = 0, 0
        if action == Action.UP:
            dx, dy = 0, -1
            self.direction = action.value
        elif action == Action.DOWN:
            dx, dy = 0, 1
            self.direction = action.value
        elif action == Action.LEFT:
            dx, dy = -1, 0
            self.direction = action.value
        elif action == Action.RIGHT:
            dx, dy = 1, 0
            self.direction = action.value
        elif action == Action.NO_ACTION:
            dx, dy = 0, 0
        elif action == Action.PLANT_BOMB and self.bomb_limit > 0:
            bomb = self.plant_bomb(grid)
            bombs.append(bomb)
            grid[bomb.pos_x][bomb.pos_y] = Tile.BOMB
        elif action == Action.PLANT_BOMB and self.bomb_limit == 0:
            return False

        step = grid[self.pos_x + dx][self.pos_y + dy]
        if step == Tile.SOLID or step == Tile.BOX or step == Tile.BOMB:
            return False
        else:
            self.pos_x += dx
            self.pos_y += dy

        return True

    def plant_bomb(self, grid):
        b = Bomb(self.bomb_range, self.pos_x, self.pos_y, grid, self, self.speed)
        self.bomb_limit -= 1
        return b

    def check_death(self, explosions: list[Explosion]):
        for explosion in explosions:
            for sector in explosion.sectors:
                if self.pos_x == sector[0] and self.pos_y == sector[1] and self.alive:
                    self.alive = False
                    return explosion.bomber

    def consume_power_up(self, power_up, power_ups):
        if power_up.type == PowerUpType.BOMB:
            self.bomb_limit += 1
        elif power_up.type == PowerUpType.FIRE:
            self.bomb_range += 1

        power_ups.remove(power_up)

    def load_animations(self, image_path, scale: float):
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

        self.animation.append(back)
        self.animation.append(right)
        self.animation.append(front)
        self.animation.append(left)
