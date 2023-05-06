from enum import Enum


class Tile(Enum):

    GROUND = 0
    SOLID = 1
    BOX = 2
    POWER_UP = 3
    PLAYER_OCCUPIED = 4
    BOMB = 5
    BOMB_ABOUT_TO_EXPLODE = 6
    BOMB_FIRE = 7

