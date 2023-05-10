import pygame
import math

from src.game.agent import Agent
from src.game.bomb import Bomb
from src.game.enums.power_up_type import PowerUpType
from src.game.enums.action import Action
from src.game.enums.tile import Tile


class Player(Agent):
    def __init__(self, x: int, y: int, speed: float):
        super().__init__(x, y, speed)
        self.pos_x = x
        self.pos_y = y
        self.speed = speed
