import numpy as np
import pygame

from src.game.power_up import PowerUp
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

    def choose_move(self,
                    grid: np.ndarray[Tile],
                    bombs: list[Bomb],
                    explosions: list[Explosion],
                    agents: list,
                    power_ups: list[PowerUp],
                    state: str):
        raise NotImplementedError

    def move(self, action: Action,
             grid: np.ndarray[Tile],
             bombs: list[Bomb],
             enemies: list,
             power_ups: list[PowerUp]) -> bool:
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
            return True
        elif action == Action.PLANT_BOMB:
            if self.bomb_limit > 0:
                bomb = self.plant_bomb(grid)
                bombs.append(bomb)
                grid[bomb.pos_x][bomb.pos_y] = Tile.BOMB
            return True

        # prevent passing through enemies
        for enemy in enemies:
            if enemy == self or not enemy.alive:
                continue
            if enemy.pos_x == self.pos_x + dx and enemy.pos_y == self.pos_y + dy:
                return False

        # make step
        step = grid[self.pos_x + dx][self.pos_y + dy]
        if step == Tile.SOLID or step == Tile.BOX or step == Tile.BOMB: # cant pass through obstacles
            return False
        else: # move
            self.pos_x += dx
            self.pos_y += dy

        # pick up power ups
        for pu in power_ups:
            if pu.pos_x == self.pos_x and pu.pos_y == self.pos_y:
                self.consume_power_up(pu, power_ups)

        return True

    def plant_bomb(self,
                   grid: np.ndarray[Tile]) -> Bomb:
        b = Bomb(self.bomb_range, self.pos_x, self.pos_y, grid, self, self.speed)
        self.bomb_limit -= 1
        return b

    def check_death(self,
                    explosions: list[Explosion]):
        for explosion in explosions:
            for sector in explosion.sectors:
                if self.pos_x == sector[0] and self.pos_y == sector[1] and self.alive:
                    self.alive = False
                    return explosion.bomber

    def consume_power_up(self,
                         power_up: PowerUp,
                         power_ups: list[PowerUp]) -> None:
        if power_up.type == PowerUpType.BOMB:
            self.bomb_limit += 1
        elif power_up.type == PowerUpType.FIRE:
            self.bomb_range += 1

        power_ups.remove(power_up)

    def load_animations(self,
                        image_path: str,
                        scale: float) -> None:
        front, back, left, right = [], [], [], []
        resize_width, resize_height = scale, scale

        front.append(pygame.transform.scale(pygame.image.load(image_path + 'f0.png'), (resize_width, resize_height)))
        front.append(pygame.transform.scale(pygame.image.load(image_path + 'f1.png'), (resize_width, resize_height)))
        front.append(pygame.transform.scale(pygame.image.load(image_path + 'f2.png'), (resize_width, resize_height)))

        right.append(pygame.transform.scale(pygame.image.load(image_path + 'r0.png'), (resize_width, resize_height)))
        right.append(pygame.transform.scale(pygame.image.load(image_path + 'r1.png'), (resize_width, resize_height)))
        right.append(pygame.transform.scale(pygame.image.load(image_path + 'r2.png'), (resize_width, resize_height)))

        back.append(pygame.transform.scale(pygame.image.load(image_path + 'b0.png'), (resize_width, resize_height)))
        back.append(pygame.transform.scale(pygame.image.load(image_path + 'b1.png'), (resize_width, resize_height)))
        back.append(pygame.transform.scale(pygame.image.load(image_path + 'b2.png'), (resize_width, resize_height)))

        left.append(pygame.transform.scale(pygame.image.load(image_path + 'l0.png'), (resize_width, resize_height)))
        left.append(pygame.transform.scale(pygame.image.load(image_path + 'l1.png'), (resize_width, resize_height)))
        left.append(pygame.transform.scale(pygame.image.load(image_path + 'l2.png'), (resize_width, resize_height)))

        self.animation.append(back)
        self.animation.append(right)
        self.animation.append(front)
        self.animation.append(left)
