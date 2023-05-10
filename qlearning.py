import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from tqdm import tqdm

from src.qlearning.QGameLoop import GameLoop



if __name__ == "__main__":
    pygame.init()

    epsilon = 0.1
    de = 0.01
    discount = 0.98
    lr = 0.01
    epochs = 10
    episodes = 1000
    speed = 1000
    states_viewed = []

    show_game = True

    # qtable = {}
    qtable_path = (Path('.') / 'src' / 'qtable' / 'qtable.csv').resolve()
    # qtable_path './qtables/9cross_space/qtable_230509T1926.csv'
    qtable = pd.read_csv(qtable_path, index_col='State',)
    qtable = qtable.transpose().to_dict(orient='list')

    for epoch in range(epochs):
        mean_score = []
        for episode in tqdm(range(episodes), desc=f'Epoch {epoch}/{epochs}', unit='game'):
            game = GameLoop(speed, show_game=False )
            mean_reward = game.run(qtable=qtable, learning_rate=lr, discount=discount, epsilon=epsilon)
            mean_score.append(mean_reward)

        mean_reward = np.mean(mean_score)
        states_viewed.append(len(qtable))
        print(f'e: {epsilon:5.5f} - viewed states:{len(qtable):5}  - mer:{mean_reward:5.3f}')
        epsilon = max(0.0, epsilon-de)
        if show_game:
            pygame.display.init()
            INFO = pygame.display.Info()
            TILE_SIZE = int(INFO.current_h * 0.05)
            WINDOW_SIZE = (13 * TILE_SIZE, 13 * TILE_SIZE)
            surface = pygame.display.set_mode(WINDOW_SIZE)
            game = GameLoop(speed=1, show_game=True, surface=surface, tile_size=TILE_SIZE)
            mean_reward = game.run(qtable=qtable, learning_rate=lr, discount=discount, epsilon=epsilon)
            pygame.display.quit()

        qtable_df = pd.DataFrame(qtable.values(), index=qtable.keys())
        qtable_df = qtable_df.rename_axis(index='State', columns="Action")
        qtable_df.to_csv(qtable_path)
