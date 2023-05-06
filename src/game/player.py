import pygame
import math

from src.game.agent import Agent
from src.game.bomb import Bomb
from src.game.enums.power_up_type import PowerUpType
from src.game.enums.action import Action
from src.game.enums.tile import Tile


class Player(Agent):

    def __init__(self, speed, x: int, y: int, tile_size: int):
        super().__init__(x, y, tile_size, speed)
        self.tile_size = tile_size
        self.pos_x = x * self.tile_size
        self.pos_y = y * self.tile_size
        self.speed = speed
