import pygame
import pygame_menu

from src import game
from src.game import Algorithm

COLOR_BACKGROUND = (153, 153, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
FPS = 60.0
MENU_BACKGROUND_COLOR = (102, 102, 153)
MENU_TITLE_COLOR = (51, 51, 255)
WINDOW_SCALE = 0.90

pygame.display.init()
INFO = pygame.display.Info()
TILE_SIZE = int(INFO.current_h * 0.05)
WINDOW_SIZE = (13 * TILE_SIZE, 13 * TILE_SIZE)

menu_theme = pygame_menu.Theme(
    selection_color=COLOR_WHITE,
    widget_font=pygame_menu.font.FONT_BEBAS,
    title_font_size=TILE_SIZE,
    title_font_color=COLOR_BLACK,
    title_font=pygame_menu.font.FONT_BEBAS,
    widget_font_color=COLOR_BLACK,
    widget_font_size=int(TILE_SIZE * 0.7),
    background_color=MENU_BACKGROUND_COLOR,
    title_background_color=MENU_TITLE_COLOR,

)


class Menu:
    clock = None
    player_alg = Algorithm.PLAYER
    en1_alg = Algorithm.DIJKSTRA
    en2_alg = Algorithm.DFS
    en3_alg = Algorithm.DIJKSTRA
    show_path = True
    surface = pygame.display.set_mode(WINDOW_SIZE)

    def __init__(self):
        return
        # self.player = Player()
        # self.enemy1 = Enemy()
        # self.enemy2 = Enemy()
        # self.enemy3 = Enemy()
        # self.map = None

    def change_path(self, *args):
        self.show_path = args[1]

    def change_player(self, *args):
        self.player_alg = args[1]

    def change_enemy1(self, *args):
        self.en1_alg = args[1]

    def change_enemy2(self, *args):
        self.en2_alg = args[1]

    def change_enemy3(self, *args):
        self.en3_alg = args[1]

    def run_game(self):
        g = game.Game( self.show_path, self.player_alg, self.en1_alg, self.en2_alg, self.en3_alg, TILE_SIZE, 2)
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
            theme=menu_theme,
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
            theme=menu_theme,
            height=int(WINDOW_SIZE[1] * WINDOW_SCALE),
            width=int(WINDOW_SIZE[0] * WINDOW_SCALE),
            title='Options'
        )
        play_options.add.selector("Character 1", [("Player", Algorithm.PLAYER),
                                                  ("DFS", Algorithm.DFS),
                                                  ("DIJKSTRA", Algorithm.DIJKSTRA), ("None", Algorithm.NONE)],
                                  onchange=self.change_player)
        play_options.add.selector("Character 2", [("DIJKSTRA", Algorithm.DIJKSTRA),
                                                  ("DFS", Algorithm.DFS),
                                                  ("None", Algorithm.NONE)], onchange=self.change_enemy1)
        play_options.add.selector("Character 3", [("DIJKSTRA", Algorithm.DIJKSTRA),
                                                  ("DFS", Algorithm.DFS),
                                                  ("None", Algorithm.NONE)], onchange=self.change_enemy2)
        play_options.add.selector("Character 4", [("DIJKSTRA", Algorithm.DIJKSTRA),
                                                  ("DFS", Algorithm.DFS),
                                                  ("None", Algorithm.NONE)], onchange=self.change_enemy3)
        play_options.add.selector("Show path", [("Yes", True), ("No", False)], onchange=self.change_path)

        play_options.add.button('Back', pygame_menu.events.BACK)

        return play_options

    def main_menu(self):

        main_menu = pygame_menu.Menu(
            theme=menu_theme,
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
            title_font_size=TILE_SIZE,
            title_font_color=COLOR_BLACK,
            title_font=pygame_menu.font.FONT_BEBAS,
            widget_font_color=COLOR_BLACK,
            widget_font_size=int(TILE_SIZE * 0.5),
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
