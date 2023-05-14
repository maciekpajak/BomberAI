from pathlib import Path

import numpy as np
import pygame

from src.game import game
from src.game.enums import Algorithm

if __name__ == '__main__':
    COLOR_BACKGROUND = (153, 153, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    FPS = 60.0
    MENU_BACKGROUND_COLOR = (102, 102, 153)
    MENU_TITLE_COLOR = (51, 51, 255)
    WINDOW_SCALE = 0.90

    pygame.init()
    pygame.display.init()
    INFO = pygame.display.Info()
    TILE_SIZE = int(INFO.current_h * 0.05)
    WINDOW_SIZE = (13 * TILE_SIZE, 13 * TILE_SIZE)

    clock = None
    player_alg = Algorithm.PLAYER
    en1_alg = Algorithm.Q
    en2_alg = Algorithm.RANDOM
    en3_alg = Algorithm.WANDER
    show_path = True
    shuffle_positions = False
    surface = pygame.display.set_mode(WINDOW_SIZE)

    grid_path = Path('.') / 'maps' / 'standard' / 'L.csv'

    grid = np.genfromtxt(grid_path, delimiter=',')
    g = game.Game(grid=grid,
                  player_alg=player_alg, en1_alg=en1_alg, en2_alg=en2_alg, en3_alg=en3_alg,
                  scale=TILE_SIZE,
                  speed=1,
                  show_path=show_path,
                  box_density=(3,6),
                  shuffle_positions=shuffle_positions,
                  max_time=120,
                  state_type='5cross')
    g.init_sprites()
    g.run(surface)
    pygame.display.quit()
    pygame.quit()