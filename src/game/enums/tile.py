from enum import Enum


class Tile(Enum):

    GROUND = 0
    SOLID = 1
    BOX = 2
    BOMB = 3
    POWER_UP = 4
    PLAYER_OCCUPIED = 5
    BOMB_ABOUT_TO_EXPLODE = 6
    EXPLOSION = 7

