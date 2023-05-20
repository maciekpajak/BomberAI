========
BomberAI
========

AI in games project. 
Bomberman game written in python using pygame. 
 


Setup
========
1. Create virtual env with your favourite tool
2. Activate it
3. To install required dependencies (based on your system) run

.. code:: sh

    python install.py

Run
========

Run GUI:

.. code:: sh

    python menu.py

Run game:

.. code:: sh

    python play.py

Bots
========

Random
------

DFS bot
------

Q-learning based bot
------

Train bot based on Q-learning:

.. code:: sh
python qlearning.py

Model compile:

.. code:: python

    def compile(self,get_reward: Callable[[bool, Action, bool, bool, int, int], float],
            state_type: str,state_range: int, min_enemy_dist: int, learning_rate=0.1,
            discount=0.98, epsilon=0.1, de=0.01, gamma=0.9, n_past_states=10) -> None:

Game settings
You can change game settings during training 

.. code:: python

    def set_game(self, grid: np.ndarray, en1_alg: Algorithm, en2_alg: Algorithm, en3_alg: Algorithm,
            box_density: int | Tuple[int, int] = 5, shuffle_positions: bool = True, max_playing_time=120) -> None:

#### Fit

.. code:: python

    def fit(self, epochs: int = 10, episodes: int = 1000, start_epoch: int = 0, show_game: bool = False,
            path_to_save: str = 'qtable.csv', log_file: str = 'log.csv') -> pd.DataFrame:


Deep-Q network based bot
------
Train bot based on deep-Q network:

.. code:: sh

    python qnetwork.py
 

Credits
=======
 
Sprites: https://opengameart.org/content/bomb-party-the-complete-set

Game mechanics based on: https://github.com/Forestf90/Bomberman