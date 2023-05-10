import numpy as np
import pandas
import pandas as pd
import pygame
import sys
import random

from src.game import Bomb, PowerUp, Tile
from src.game.enums import Action
from src.game.player import Player
from src.game.explosion import Explosion
from src.game.enemy import Enemy
from src.game.enums.algorithm import Algorithm

BACKGROUND_COLOR = (107, 142, 35)

# font = None
#
# player = None
# enemy_list = []
# ene_blocks = []
# bombs = []
# explosions = []
# power_ups = []

GRID_BASE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


# GRID_BASE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


class Game:

    def __init__(self, show_path: bool, player_alg, en1_alg, en2_alg, en3_alg, scale, speed):
        self.enemy_list: list[Enemy] = []
        self.ene_blocks = []
        self.explosions: list[Explosion] = []
        self.bombs: list[Bomb] = []
        self.power_ups: list[PowerUp] = []
        self.game_ended = False
        self.grid = None
        self.player = Player(speed, 1, 1, 4)
        self.show_path = show_path
        self.scale = scale
        self.speed = speed

        self.init_players(player_alg, en1_alg, en2_alg, en3_alg)

    def init_players(self, player_alg, en1_alg, en2_alg, en3_alg):
        if en1_alg is not Algorithm.NONE:
            en1 = Enemy(11, 11, en1_alg, self.speed)
            en1.load_animations('1', self.scale)
            self.enemy_list.append(en1)
            self.ene_blocks.append(en1)

        if en2_alg is not Algorithm.NONE:
            en2 = Enemy(1, 11, en2_alg, self.speed)
            en2.load_animations('2', self.scale)
            self.enemy_list.append(en2)
            self.ene_blocks.append(en2)

        if en3_alg is not Algorithm.NONE:
            en3 = Enemy(11, 1, en3_alg, self.speed)
            en3.load_animations('3', self.scale)
            self.enemy_list.append(en3)
            self.ene_blocks.append(en3)

        if player_alg is Algorithm.PLAYER:
            self.player.load_animations('images/hero/p', self.scale)
            self.ene_blocks.append(self.player)
        elif player_alg is not Algorithm.NONE:
            en0 = Enemy(1, 1, player_alg, self.speed)
            en0.load_animations('', self.scale)
            self.enemy_list.append(en0)
            self.ene_blocks.append(en0)
            self.player.life = False
        else:
            self.player.life = False

    def init_sprites(self):
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

    def draw(self, surface):
        surface.fill(BACKGROUND_COLOR)
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                surface.blit(self.terrain_images[self.grid[i][j]],
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
        if self.player.life:
            surface.blit(self.player.animation[self.player.direction][self.player.frame], (
                self.player.pos_x * (self.scale / 4), self.player.pos_y * (self.scale / 4), self.scale, self.scale))
        for en in self.enemy_list:
            if en.life:
                surface.blit(en.animation[en.direction][en.frame],
                                  (en.pos_x * (self.scale / 4), en.pos_y * (self.scale / 4), self.scale, self.scale))
                if self.show_path:
                    if en.algorithm == Algorithm.DFS:
                        for sek in en.path:
                            pygame.draw.rect(surface, (255, 0, 0, 240),
                                             [sek[0] * self.scale, sek[1] * self.scale, self.scale, self.scale], 1)
                    else:
                        for sek in en.path:
                            pygame.draw.rect(surface, (255, 0, 255, 240),
                                             [sek[0] * self.scale, sek[1] * self.scale, self.scale, self.scale], 1)

        if self.game_ended:
            font = pygame.font.SysFont('Bebas', self.scale)
            tf = font.render("Press ESC to go back to menu", False, (153, 153, 255))
            surface.blit(tf, (10, 10))

        pygame.display.update()

    def run(self, surface):
        # self.grid = [row[:] for row in GRID_BASE]
        self.generate_map(GRID_BASE)
        # power_ups.append(PowerUp(1, 2, PowerUpType.BOMB))
        # power_ups.append(PowerUp(2, 1, PowerUpType.FIRE))
        clock = pygame.time.Clock()

        running = True
        game_ended = False
        while running:
            dt = clock.tick(int(15 * self.speed))
            for en in self.enemy_list:
                en.make_move(self.grid, self.bombs, self.explosions, self.ene_blocks)

            # print(self.player.state)
            if self.player.life:
                keys = pygame.key.get_pressed()
                action = Action.NO_ACTION
                if keys[pygame.K_DOWN]:
                    action = Action.DOWN
                elif keys[pygame.K_RIGHT]:
                    action = Action.RIGHT
                elif keys[pygame.K_UP]:
                    action = Action.UP
                elif keys[pygame.K_LEFT]:
                    action = Action.LEFT
                else:
                    action = Action.NO_ACTION
                self.move_player(action)
            self.draw(surface)
            # print(pd.DataFrame(self.grid))
            # print(self.get_state())
            # print(self.get_9grid_state())
            # print(self.get_9crossgrid_state())
            # print(self.get_5grid_state())

            if not game_ended:
                game_ended = self.check_end_game()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        self.move_player(Action.PLANT_BOMB)

            self.update_bombs(dt)

        self.explosions.clear()
        self.enemy_list.clear()
        self.ene_blocks.clear()
        self.power_ups.clear()

    # def display(self):
    def move_player(self, action):
        if action == Action.PLANT_BOMB:
            if self.player.bomb_limit > 0 and self.player.life:
                temp_bomb = self.player.plant_bomb(self.grid)
                self.bombs.append(temp_bomb)
                self.grid[temp_bomb.pos_x][temp_bomb.pos_y] = 3
                self.player.bomb_limit -= 1
        else:
            self.player.move(action, self.grid, self.ene_blocks, self.power_ups)

    def update_bombs(self, dt):
        sectors_cleared_by_player = None
        sectors_cleared = 0
        player_killed_enemy = False
        for b in self.bombs:
            b.update(dt)
            self.grid[b.pos_x][b.pos_y] = 3
            if b.time_to_explode < 1:
                b.bomber.bomb_limit += 1
                self.grid[b.pos_x][b.pos_y] = 0
                exp_temp = Explosion(b.pos_x, b.pos_y, b.range, self.speed)
                exp_temp.explode(self.grid, self.bombs, b, self.power_ups)
                sectors_cleared = exp_temp.clear_sectors(self.grid, self.power_ups)
                self.explosions.append(exp_temp)
                if b.bomber == self.player:
                    sectors_cleared_by_player = sectors_cleared
        if self.player not in self.enemy_list:
            self.player.check_death(self.explosions)
        for en in self.enemy_list:
            bomber = en.check_death(self.explosions)
            if bomber == self.player:
                print("Player killed enemy")
                player_killed_enemy = True
        for e in self.explosions:
            e.update(dt)
            if e.time < 1:
                self.explosions.remove(e)
        return player_killed_enemy, sectors_cleared_by_player

    def check_end_game(self):
        if not self.player.life:
            return True

        for en in self.enemy_list:
            if en.life:
                return False

        return True

    def generate_map(self, grid_ori):
        self.grid = np.copy(grid_ori)
        for i in range(1, len(self.grid) - 1):
            for j in range(1, len(self.grid[i]) - 1):
                if self.grid[i][j] != 0:
                    continue
                elif (i < 4 or i > len(self.grid) - 5) and (j < 4 or j > len(self.grid[i]) - 5):
                    continue
                if True:  # random.randint(0, 9) < 4:
                    self.grid[i][j] = 2

        self.grid = self.grid

    def get_state(self):
        state = ''.join([str(y_show) for x_show in self.grid for y_show in x_show])
        state += ''.join([str(self.player.pos_x).zfill(2), str(self.player.pos_y).zfill(2)])
        enemy_state = ''
        for enemy in self.enemy_list:
            state += ''.join([str(enemy.pos_x).zfill(2), str(enemy.pos_y).zfill(2)])
        state += enemy_state.zfill(4 * 2 * 2)
        bombs_state = ''
        for bomb in self.bombs[:10]:
            bombs_state += ''.join([str(bomb.pos_x).zfill(2), str(bomb.pos_y).zfill(2)])
        state += bombs_state.zfill(10 * 2 * 2)
        return state

    def get_9grid_state(self):
        x, y = int(self.player.pos_x / 4), int(self.player.pos_y / 4)
        state = ''.join([str(self.grid[x + i][y + j]) for i in [-1, 0, 1] for j in [-1, 0, 1]])
        enemy_dist = 10000
        for enemy in self.enemy_list:
            dist = abs(enemy.pos_y - y) + abs(enemy.pos_x - x)
            enemy_dist = min(enemy_dist, dist)
        state += str(enemy_dist).zfill(3)
        bombs_state = '0'
        for bomb in self.bombs:
            if x - 1 <= bomb.pos_x <= x + 1 and y - 1 <= bomb.pos_y <= y + 1:
                bombs_state = str(bomb.frame)
        state += bombs_state
        return state

    def get_9crossgrid_state(self):
        state = 'g'
        x, y = int(self.player.pos_x / 4), int(self.player.pos_y / 4)
        tiles5 = [[x, y], [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1], [x + 2, y], [x - 2, y], [x, y + 2],
                  [x, y - 2]]
        for tile in tiles5:
            if tile[0] < 0 or tile[0] >= len(self.grid) or tile[1] < 0 or tile[1] >= len(self.grid):
                state += '1'
            else:
                state += str(self.grid[tile[0]][tile[1]])

        for tile in tiles5:
            bombs_state = 'b00'
            for bomb in self.bombs:
                if bomb.pos_x == tile[0] and bomb.pos_y == tile[1]:
                    bombs_state = 'b1' + str(bomb.frame + 1)
            state += bombs_state

        enemy_dist = 20
        y_prefix = 0
        dist_y = 0
        x_prefix = 0
        dist_x = 0
        for enemy in self.enemy_list:
            dist = abs(enemy.pos_y - y * 4) + abs(enemy.pos_x - x * 4)
            if dist < enemy_dist:
                enemy_dist = dist
                y_prefix = 0 if enemy.pos_y - y * 4 >= 0 else 1
                x_prefix = 0 if enemy.pos_x - x * 4 >= 0 else 1
                dist_y = abs(enemy.pos_y - y * 4)
                dist_x = abs(enemy.pos_x - x * 4)
        enemy_state = ''.join(["ex", str(x_prefix), str(dist_x).zfill(2),
                               "ey", str(y_prefix), str(dist_y).zfill(2)])
        state += enemy_state

        return state

    def get_5grid_state(self):
        state = 'g'
        x, y = int(self.player.pos_x / 4), int(self.player.pos_y / 4)
        tiles5 = [[x, y], [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]
        for tile in tiles5:
            state += str(self.grid[tile[0]][tile[1]])

        for tile in tiles5:
            bombs_state = 'b00'
            for bomb in self.bombs:
                if bomb.pos_x == tile[0] and bomb.pos_y == tile[1]:
                    bombs_state = 'b1' + str(bomb.frame + 1)
            state += bombs_state

        enemy_dist = 20
        y_prefix = 0
        dist_y = 0
        x_prefix = 0
        dist_x = 0
        for enemy in self.enemy_list:
            dist = abs(enemy.pos_y - y * 4) + abs(enemy.pos_x - x * 4)
            if dist < enemy_dist:
                enemy_dist = dist
                y_prefix = 0 if enemy.pos_y - y * 4 >= 0 else 1
                x_prefix = 0 if enemy.pos_x - x * 4 >= 0 else 1
                dist_y = abs(enemy.pos_y - y * 4)
                dist_x = abs(enemy.pos_x - x * 4)
        enemy_state = ''.join(["ex", str(x_prefix), str(dist_x).zfill(2),
                               "ey", str(y_prefix), str(dist_y).zfill(2)])
        state += enemy_state

        return state
