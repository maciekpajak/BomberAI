import os
from pathlib import Path

import numpy as np
import pygame
import pygame_menu

from src.game import game
from src.game import Algorithm

COLOR_BACKGROUND = (153, 153, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
FPS = 60.0
MENU_BACKGROUND_COLOR = (102, 102, 153)
MENU_TITLE_COLOR = (51, 51, 255)
WINDOW_SCALE = 0.90
WINDOW_SIZE = (500, 500)


class Menu:

    def __init__(self):

        pygame.display.init()
        self.INFO = pygame.display.Info()
        self.TILE_SIZE = int(self.INFO.current_h * 0.05)

        self.menu_theme = pygame_menu.Theme(
            selection_color=COLOR_WHITE,
            widget_font=pygame_menu.font.FONT_BEBAS,
            title_font_size=self.TILE_SIZE,
            title_font_color=COLOR_BLACK,
            title_font=pygame_menu.font.FONT_BEBAS,
            widget_font_color=COLOR_BLACK,
            widget_font_size=int(self.TILE_SIZE * 0.7),
            background_color=MENU_BACKGROUND_COLOR,
            title_background_color=MENU_TITLE_COLOR,

        )

        self.clock = None
        self.grid = np.genfromtxt('maps/standard/L.csv', delimiter=',')
        self.player_alg = Algorithm.PLAYER
        self.en1_alg = Algorithm.DFS
        self.en2_alg = Algorithm.DFS
        self.en3_alg = Algorithm.DFS

        self.show_path = False
        self.shuffle_positions = False
        self.box_density = 5
        self.max_playing_time = 120

        self.surface = pygame.display.set_mode(WINDOW_SIZE)

    def change_player(self, name, alg):
        self.player_alg = alg

    def change_enemy1(self, name, alg):
        self.en1_alg = alg

    def change_enemy2(self, name, alg):
        self.en2_alg = alg

    def change_enemy3(self, name, alg):
        self.en3_alg = alg

    def change_map(self, name, path):
        self.grid = np.genfromtxt(path, delimiter=',')

    def change_box_density(self, density):
        self.box_density = density

    def change_shuffle(self, shuffle):
        self.shuffle_positions = shuffle

    def change_path(self, show):
        self.show_path = show

    def run_game(self):
        g = game.Game(grid=self.grid,
                      player_alg=self.player_alg,
                      en1_alg=self.en1_alg,
                      en2_alg=self.en2_alg,
                      en3_alg=self.en3_alg,
                      scale=WINDOW_SIZE[0] / len(self.grid),
                      speed=1,
                      show_path=self.show_path,
                      box_density=self.box_density,
                      shuffle_positions=self.shuffle_positions,
                      max_playing_time=self.max_playing_time)
        g.init_sprites()
        g.run(self.surface)

    def main_background(self):
        self.surface.fill(COLOR_BACKGROUND)

    def menu_loop(self):
        pygame.init()

        pygame.display.set_caption('Bomberman')
        self.clock = pygame.time.Clock()

        running = True
        main_menu = self.main_menu()
        while running:

            self.clock.tick(FPS)

            self.main_background()

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False

            if main_menu.is_enabled():
                main_menu.mainloop(self.surface, self.main_background)

            pygame.display.flip()

        exit()

    def play_menu(self):

        play_menu = pygame_menu.Menu(
            theme=self.menu_theme,
            height=int(WINDOW_SIZE[1] * WINDOW_SCALE),
            width=int(WINDOW_SIZE[0] * WINDOW_SCALE),
            title='Play menu'
        )

        play_menu.add.button('Start', action=self.run_game)
        play_menu.add.button('Options', action=self.play_options())
        play_menu.add.button('Return  to  main  menu', action=pygame_menu.events.BACK)

        return play_menu

    def play_options(self):
        play_options = pygame_menu.Menu(
            theme=self.menu_theme,
            height=int(WINDOW_SIZE[1] * WINDOW_SCALE),
            width=int(WINDOW_SIZE[0] * WINDOW_SCALE),
            title='Options'
        )
        play_options.add.selector(title="Character 1",
                                  items=[(alg.name, alg) for alg in Algorithm],
                                  default=[alg for alg in Algorithm].index(Algorithm.PLAYER),
                                  onchange=self.change_player)
        play_options.add.selector(title="Character 2",
                                  items=[(alg.name, alg) for alg in Algorithm if alg != Algorithm.PLAYER],
                                  default=[alg for alg in Algorithm].index(Algorithm.DFS),
                                  onchange=self.change_enemy1)
        play_options.add.selector(title="Character 3",
                                  items=[(alg.name, alg) for alg in Algorithm if alg != Algorithm.PLAYER],
                                  default=[alg for alg in Algorithm].index(Algorithm.DFS),
                                  onchange=self.change_enemy2)
        play_options.add.selector(title="Character 4",
                                  items=[(alg.name, alg) for alg in Algorithm if alg != Algorithm.PLAYER],
                                  default=[alg for alg in Algorithm].index(Algorithm.DFS),
                                  onchange=self.change_enemy3)
        play_options.add.selector(title="Map",
                                  items=[(Path(root, name).with_suffix('').name, Path(root, name))
                                         for root, dirs, files in os.walk("maps", topdown=False)
                                         for name in files],
                                  onchange=self.change_map)
        play_options.add.range_slider("Box density",
                                      range_values=[i + 1 for i in range(10)],
                                      default=5,
                                      onchange=self.change_box_density)
        play_options.add.toggle_switch("Show path", default=0, onchange=self.change_path)
        play_options.add.toggle_switch("Shuffle positions", default=0, onchange=self.change_shuffle)

        play_options.add.button('Back', pygame_menu.events.BACK)

        return play_options

    def main_menu(self):

        main_menu = pygame_menu.Menu(
            theme=self.menu_theme,
            height=int(WINDOW_SIZE[1] * WINDOW_SCALE),
            width=int(WINDOW_SIZE[0] * WINDOW_SCALE),
            onclose=pygame_menu.events.EXIT,
            title='Main menu'
        )

        main_menu.add.button('Play', self.play_menu())
        main_menu.add.button('About', self.about_menu())
        main_menu.add.button('Quit', pygame_menu.events.EXIT)

        return main_menu

    def about_menu(self):
        theme = pygame_menu.themes.Theme(
            selection_color=COLOR_WHITE,
            widget_font=pygame_menu.font.FONT_BEBAS,
            title_font_size=self.TILE_SIZE,
            title_font_color=COLOR_BLACK,
            title_font=pygame_menu.font.FONT_BEBAS,
            widget_font_color=COLOR_BLACK,
            widget_font_size=int(self.TILE_SIZE * 0.5),
            background_color=MENU_BACKGROUND_COLOR,
            title_background_color=MENU_TITLE_COLOR
        )

        about_menu = pygame_menu.Menu(
            theme=theme,
            height=int(WINDOW_SIZE[1] * WINDOW_SCALE),
            width=int(WINDOW_SIZE[0] * WINDOW_SCALE),
            overflow=False,
            title='About'
        )
        about_menu.add.label("Player controls: ")
        about_menu.add.label("Movement: Arrows")
        about_menu.add.label("Plant bomb: Space")
        about_menu.add.label("Credits: Michal Sliwa")
        about_menu.add.label("Sprite: ")
        about_menu.add.label("https://opengameart.org/ content/bomb-party-the-complete-set", wordwrap=True)
        about_menu.add.vertical_margin(25)
        about_menu.add.button('Return  to  main  menu', pygame_menu.events.BACK)

        return about_menu


if __name__ == "__main__":
    menu = Menu()
    menu.menu_loop()
