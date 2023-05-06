import time

import numpy as np
import pandas as pd
import pygame

from src.qlearning.QGameLoop import GameLoop



if __name__ == "__main__":
    pygame.init()

    mean_score = []
    epsilon = 0.01
    de = 0.00001
    discount = 0.98
    lr = 0.01
    epochs = 1000
    qtable = {}
    speed = 100
    states_viewed = []

    show_game = True

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs}', end=' ')
        if epoch % 100 == 0:
            pygame.display.init()
            INFO = pygame.display.Info()
            TILE_SIZE = int(INFO.current_h * 0.05)
            WINDOW_SIZE = (13 * TILE_SIZE, 13 * TILE_SIZE)
            surface = pygame.display.set_mode(WINDOW_SIZE)
            game = GameLoop(speed=3, show_game=True, surface=surface, tile_size=TILE_SIZE)
            mean_reward = game.run(qtable=qtable, learning_rate=lr, discount=discount, epsilon=epsilon)
            pygame.display.quit()
        else:
            game = GameLoop(speed, show_game=False )
            mean_reward = game.run(qtable=qtable, learning_rate=lr, discount=discount, epsilon=epsilon)

        mean_score.append(mean_reward)
        epsilon = max(0.0, epsilon-de)
        states_viewed.append(len(qtable))
        print(f'e: {epsilon:5.5f} - viewed states:{len(qtable):5}  - mer:{mean_reward:5.3f}')


    qtable_df = pd.DataFrame(qtable.values(), index=qtable.keys())
    qtable_df = qtable_df.rename_axis(index='State', columns="Action")
    qtable_df.to_csv(f'./qtables/9cross_space/qtable_epochs{epochs}_lr{lr}.csv')
