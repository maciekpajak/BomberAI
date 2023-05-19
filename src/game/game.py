import time
from typing import Tuple

import numpy as np
import pygame
import sys
import random

from src.game import Bomb, PowerUp, Tile, Agent
from src.game.enums import Action
from src.game.player import Player
from src.game.explosion import Explosion
from src.game.enemy import Enemy
from src.game.enums.algorithm import Algorithm

BACKGROUND_COLOR = (107, 142, 35)


class Game:

    def __init__(self, grid: np.ndarray,
                 player_alg: Algorithm,
                 en1_alg: Algorithm,
                 en2_alg: Algorithm,
                 en3_alg: Algorithm,
                 scale: float,
                 speed: float = 1,
                 show_path: bool = False,
                 box_density: int | Tuple[int, int] = 5,
                 shuffle_positions: bool = True,
                 max_time: int = 120,
                 state_type: str = '5cross'):
        self.enemy_list: list[Enemy] = []
        self.agents_on_board: list[Agent] = []
        self.explosions: list[Explosion] = []
        self.bombs: list[Bomb] = []
        self.power_ups: list[PowerUp] = []
        self.game_ended: bool = False
        self.grid_h, self.grid_w = self.generate_map(grid, box_density=box_density)
        self.show_path = show_path
        self.scale: float = scale
        self.speed: float = speed
        self.playing_time = 0
        self.max_time = max_time / speed
        self.state_type = state_type
        self.min_enemy_dist = 10
        self.player = None
        self.shuffle_positions = shuffle_positions
        self.init_players(player_alg, en1_alg, en2_alg, en3_alg, shuffle_positions)

    def init_players(self,
                     player_alg: Algorithm,
                     en1_alg: Algorithm,
                     en2_alg: Algorithm,
                     en3_alg: Algorithm,
                     shuffle_positions: bool = True) -> None:
        left_up = [1, 1]
        left_bottom = [self.grid_h - 2, 1]
        right_up = [1, self.grid_w - 2]
        right_bottom = [self.grid_h - 2, self.grid_w - 2]
        pos = [right_bottom, right_up, left_bottom, left_up]
        if shuffle_positions:
            np.random.shuffle(pos)
        if en1_alg is not Algorithm.NONE:
            en1 = Enemy(pos[0][0], pos[0][1], en1_alg, self.speed)
            en1.load_animations('images/enemy/e1', self.scale)
            self.enemy_list.append(en1)
            self.agents_on_board.append(en1)

        if en2_alg is not Algorithm.NONE:
            en2 = Enemy(pos[1][0], pos[1][1], en2_alg, self.speed)
            en2.load_animations('images/enemy/e2', self.scale)
            self.enemy_list.append(en2)
            self.agents_on_board.append(en2)

        if en3_alg is not Algorithm.NONE:
            en3 = Enemy(pos[2][0], pos[2][1], en3_alg, self.speed)
            en3.load_animations('images/enemy/e3', self.scale)
            self.enemy_list.append(en3)
            self.agents_on_board.append(en3)

        if player_alg is Algorithm.PLAYER:
            self.player = Player(pos[3][0], pos[3][1], self.speed)
            self.player.load_animations('images/hero/p', self.scale)
            self.agents_on_board.append(self.player)
        elif player_alg is not Algorithm.NONE:
            en0 = Enemy(pos[3][0], pos[3][1], player_alg, self.speed)
            en0.load_animations('images/hero/p', self.scale)
            self.enemy_list.append(en0)
            self.agents_on_board.append(en0)

    def init_sprites(self) -> None:
        self.grass_img = pygame.image.load('images/terrain/grass.png')
        self.grass_img = pygame.transform.scale(self.grass_img, (self.scale, self.scale))

        self.block_img = pygame.image.load('images/terrain/block.png')
        self.block_img = pygame.transform.scale(self.block_img, (self.scale, self.scale))

        self.box_img = pygame.image.load('images/terrain/box.png')
        self.box_img = pygame.transform.scale(self.box_img, (self.scale, self.scale))

        self.bomb1_img = pygame.image.load('images/bomb/1.png')
        self.bomb1_img = pygame.transform.scale(self.bomb1_img, (self.scale, self.scale))

        self.bomb2_img = pygame.image.load('images/bomb/2.png')
        self.bomb2_img = pygame.transform.scale(self.bomb2_img, (self.scale, self.scale))

        self.bomb3_img = pygame.image.load('images/bomb/3.png')
        self.bomb3_img = pygame.transform.scale(self.bomb3_img, (self.scale, self.scale))

        self.explosion1_img = pygame.image.load('images/explosion/1.png')
        self.explosion1_img = pygame.transform.scale(self.explosion1_img, (self.scale, self.scale))

        self.explosion2_img = pygame.image.load('images/explosion/2.png')
        self.explosion2_img = pygame.transform.scale(self.explosion2_img, (self.scale, self.scale))

        self.explosion3_img = pygame.image.load('images/explosion/3.png')
        self.explosion3_img = pygame.transform.scale(self.explosion3_img, (self.scale, self.scale))

        self.terrain_images = [self.grass_img, self.block_img, self.box_img, self.grass_img]
        self.bomb_images = [self.bomb1_img, self.bomb2_img, self.bomb3_img]
        self.explosion_images = [self.explosion1_img, self.explosion2_img, self.explosion3_img]

        self.power_up_bomb_img = pygame.image.load('images/power_up/bomb.png')
        self.power_up_bomb_img = pygame.transform.scale(self.power_up_bomb_img, (self.scale, self.scale))

        self.power_up_fire_img = pygame.image.load('images/power_up/fire.png')
        self.power_up_fire_img = pygame.transform.scale(self.power_up_fire_img, (self.scale, self.scale))

        self.power_ups_images = [self.power_up_bomb_img, self.power_up_fire_img]

    def draw(self, surface: pygame.Surface | pygame.SurfaceType):
        surface.fill(BACKGROUND_COLOR)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                surface.blit(self.terrain_images[self.grid[i][j].value],
                             (i * self.scale, j * self.scale, self.scale, self.scale))

        for pu in self.power_ups:
            surface.blit(self.power_ups_images[pu.type.value],
                         (pu.pos_x * self.scale, pu.pos_y * self.scale, self.scale, self.scale))

        for x in self.bombs:
            surface.blit(self.bomb_images[x.frame],
                         (x.pos_x * self.scale, x.pos_y * self.scale, self.scale, self.scale))

        for y in self.explosions:
            for x in y.sectors:
                surface.blit(self.explosion_images[y.frame],
                             (x[0] * self.scale, x[1] * self.scale, self.scale, self.scale))
        if self.player is not None and self.player.alive:
            surface.blit(self.player.animation[self.player.direction][self.player.frame], (
                self.player.pos_x * self.scale, self.player.pos_y * self.scale, self.scale, self.scale))
        for en in self.enemy_list:
            if en.alive:
                surface.blit(en.animation[en.direction][en.frame],
                             (en.pos_x * self.scale, en.pos_y * self.scale, self.scale, self.scale))
                if self.show_path:
                    if en.algorithm == Algorithm.DFS:
                        for sek in en.path:
                            pygame.draw.rect(surface, (255, 0, 0, 240),
                                             [sek[0] * self.scale, sek[1] * self.scale, self.scale, self.scale], 1)
                    else:
                        for sek in en.path:
                            pygame.draw.rect(surface, (255, 0, 255, 240),
                                             [sek[0] * self.scale, sek[1] * self.scale, self.scale, self.scale], 1)

        if not self.game_ended:
            font = pygame.font.SysFont('Bebas', int(self.scale))
            time_left = self.max_time - self.playing_time
            tf = font.render(f"{int(time_left)} s", False, (153, 153, 255))
            surface.blit(tf, (10, 10))

        if self.game_ended:
            font = pygame.font.SysFont('Bebas', int(self.scale))
            tf = font.render("Press ESC to go back to menu", False, (153, 153, 255))
            surface.blit(tf, (10, 10))

        pygame.display.update()

    def run(self, surface: pygame.Surface | pygame.SurfaceType):
        clock = pygame.time.Clock()

        running = True
        self.draw(surface)
        start_time = time.time()
        while running:
            dt = clock.tick(15 * self.speed)

            action = Action.NO_ACTION
            if self.player is not None and self.player.alive:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_DOWN]:
                    action = Action.DOWN
                elif keys[pygame.K_RIGHT]:
                    action = Action.RIGHT
                elif keys[pygame.K_UP]:
                    action = Action.UP
                elif keys[pygame.K_LEFT]:
                    action = Action.LEFT

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        action = Action.PLANT_BOMB

            if self.player is not None:
                self.player.move(action, self.grid, self.bombs, self.agents_on_board, self.power_ups)
            for en in self.enemy_list:
                state = self.get_state(agent=en)
                en.choose_move(self.grid, self.bombs, self.explosions, self.agents_on_board, self.power_ups, state)

            self.update_bombs(dt)

            self.draw(surface)

            if not self.game_ended:
                self.playing_time = time.time() - start_time
                self.game_ended = self.check_end_game()

        self.explosions.clear()
        self.enemy_list.clear()
        self.agents_on_board.clear()
        self.power_ups.clear()

    def update_bombs(self, dt: float) -> Tuple[bool, int]:
        sectors_cleared_by_player = None
        player_killed_enemy = False
        for bomb in self.bombs:
            bomb.update(dt)
            self.grid[bomb.pos_x][bomb.pos_y] = Tile.BOMB
            if bomb.time_to_explode <= 0:
                bomb.bomber.bomb_limit += 1
                self.grid[bomb.pos_x][bomb.pos_y] = Tile.GROUND
                exp_temp = Explosion(bomb.pos_x, bomb.pos_y, bomb.range, self.speed)
                exp_temp.explode(self.grid, self.bombs, bomb, self.power_ups)
                sectors_cleared = exp_temp.clear_sectors(self.grid, self.power_ups)
                self.explosions.append(exp_temp)
                if bomb.bomber == self.player:
                    sectors_cleared_by_player = sectors_cleared
        for agent in self.agents_on_board:
            bomber = agent.check_death(self.explosions)
            if agent != self.player and bomber == self.player:
                player_killed_enemy = True
        for e in self.explosions:
            e.update(dt)
            if e.time < 1:
                self.explosions.remove(e)
        return player_killed_enemy, sectors_cleared_by_player

    def check_end_game(self)->bool:
        if self.playing_time > self.max_time:
            return True

        if self.player is not None and not self.player.alive:
            return True

        for en in self.enemy_list:
            if en.alive:
                return False

        return True

    def generate_map(self,
                     grid: np.ndarray[int],
                     box_density: int | Tuple[int, int] = 5,
                     no_box_area: int = 2) -> Tuple[int, int]:
        self.grid: np.ndarray[Tile] = np.empty_like(grid, dtype=Tile)
        self.grid[grid == Tile.SOLID.value] = Tile.SOLID
        self.grid[grid == Tile.GROUND.value] = Tile.GROUND
        h, w = self.grid.shape

        if isinstance(box_density, Tuple):
            box_density = random.randint(box_density[0], box_density[1])
        elif isinstance(box_density, int):
            box_density = box_density
        else:
            raise ValueError

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self.grid[y][x] == Tile.SOLID:
                    continue
                if (no_box_area < y < h - no_box_area - 1) or (no_box_area < x < w - no_box_area - 1):
                    if random.randint(0, 9) < box_density:
                        self.grid[y][x] = Tile.BOX
        return h, w

    def get_state(self, agent: Agent) -> str:

        agent_state = ''  # str(agent.bomb_limit) + str(agent.bomb_range)

        x, y = agent.pos_x, agent.pos_y
        if self.state_type == 'full':
            tiles = [[xx, yy] for xx in range(self.grid_w) for yy in range(self.grid_h)]
        elif self.state_type == '9cross':
            tiles = [[x, y], [x + 1, y], [x + 2, y], [x - 1, y], [x - 2, y], [x, y + 1], [x, y + 2], [x, y - 1],
                     [x, y - 2]]
        elif self.state_type == '9square':
            raise NotImplementedError
        elif self.state_type == '9square+cross':
            raise NotImplementedError
        elif self.state_type == '5cross':
            tiles = [[x, y], [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]
        else:
            raise NotImplementedError

        surrounding_state = ''
        for tile in tiles:
            if tile[0] < 0 or tile[0] >= len(self.grid) or tile[1] < 0 or tile[1] >= len(self.grid):
                surrounding_state += str(Tile.SOLID.value)
            else:
                surrounding_state += str(self.grid[tile[0]][tile[1]].value)

        bombs_state = ''
        for tile in tiles:
            tmp_bomb_state = '00'
            for bomb in self.bombs:
                if bomb.pos_x == tile[0] and bomb.pos_y == tile[1]:
                    tmp_bomb_state = '1' + str(bomb.frame + 1)
            bombs_state += tmp_bomb_state

        closest_enemy = agent
        for enemy in self.agents_on_board:
            if enemy == agent:
                continue
            dist = abs(enemy.pos_y - y) + abs(enemy.pos_x - x)  # manhattan dist
            if dist < self.min_enemy_dist:
                min_enemy_dist = dist
                closest_enemy = enemy

        y_prefix = 0 if closest_enemy.pos_y - y >= 0 else 1
        x_prefix = 0 if closest_enemy.pos_x - x >= 0 else 1
        dist_y = abs(closest_enemy.pos_y - y)
        dist_x = abs(closest_enemy.pos_x - x)
        closest_enemy_state = ''.join([str(x_prefix), str(dist_x).zfill(2), str(y_prefix), str(dist_y).zfill(2)])

        state = agent_state + surrounding_state + bombs_state + closest_enemy_state
        return state
