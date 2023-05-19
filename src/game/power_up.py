from src.game.enums.power_up_type import PowerUpType


class PowerUp:

    def __init__(self, x: int, y: int, power_type: PowerUpType):
        self.pos_x: int = x
        self.pos_y: int = y
        self.type: PowerUpType = power_type
