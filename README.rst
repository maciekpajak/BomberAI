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

Bots
========

Random
------

This bot picks random path, drop the bomb
and run away to the safe place.

DFS bot
------

This bot picks shortest path to the destroyable object using DFS algorithm, drop the bomb
and run away to the safe place.

Q-learning based bot
------
This bot picks the best move based on the Q table

Deep Q-learning based bot
------
This bot picks the best move using neural network


Training bots
=============

Q-learning
------

1. Write reward function

.. code:: python

    def get_reward(player_alive: bool, action: Action, is_move_possible: bool,
                    suicide: bool, kills: int, destroyed_boxes: int) -> float:
        ...
        return reward

Parameters

~~~~~~~~~~~~
* player_alive: bool
    True if the bot is alive, otherwise False
* action: Action
    The action chosen by bot in given moment
* is_move_possible: bool
    True if action is possible, otherwise False
* suicide: bool
    True if the bot killed himself with his bomb, otherwise False
* kills: int
    How many enemies the bot's bombs killed in given moment
* destroyed_boxes: int
    How many boxes the bot's bombs destroyed in given moment

2. Init model

.. code:: python

    model = Model()

3. Load Qtable (optional)

.. code:: python

    model.load(path: str)

3. Compile model

.. code:: python

    model.compile(get_reward: Callable, state_type: str,state_range: int,
                    min_enemy_dist: int, learning_rate=0.1, discount=0.98,
                    epsilon=0.1, de=0.01, gamma=0.9, n_past_states=10)


Parameters

~~~~~~~~~~~~
- get_reward: Callable
    reward function
- state_type: str
    Type of state used by bot. Possible 'full', 'circle', 'square', 'cross'
- state_range: int
    Radius of the bot surrounding shape
- min_enemy_dist: int
    Minimum distance from enemies included in state
- learning_rate: float
    Learning rate parameter
- discount: float
    Discount factor
- epsilon : float
    Exploration chance
- de : float
    Parameter to decreasing epsilon at the end of the epoch
- gamma : float
    Factor to update past states `(gamma ^ t[i]) * reward`,
    where i in [0..n_past_states]. The earlier the state,
    the less it improves.
- n_past_states: int
    How many past states to update


4. Set game

.. code:: python

    model.set_game(grid: np.ndarray[int], en1_alg: Algorithm, en2_alg: Algorithm,
                    en3_alg: Algorithm, box_density: int | Tuple[int, int] = 5,
                    shuffle_positions: bool = True, max_playing_time=120)
Parameters

~~~~~~~~~~~~
* grid: np.ndarray[int]
    A map of the maze for players - ground = 0, unbreakable wall = 1,
* en1_alg: Algorithm
    Algorithm of the first enemy
* en2_alg: Algorithm
    Algorithm of the second enemy
* en3_alg: Algorithm
    Algorithm of the third enemy
* box_density: int | Tuple[int, int]
    How densely the boxes are to be arranged on the map
* shuffle_positions: bool = True
    Whether to shuffle players' positions
* max_playing_time: int = 120
    Maximum time for gameplay


5. Fit

.. code:: python

    model.fit(epochs: int = 10, episodes: int = 1000, start_epoch: int = 0,
                show_game: bool = False, path_to_save: str = 'qtable.csv',
                log_file: str = 'log.csv')

Parameters

~~~~~~~~~~~~
- epochs: int = 10
    The number of epochs to train the model.
- episodes: int = 1000
    The number of episodes to run.
- start_epoch: int = 0
    The starting epoch for training.
- show_game: bool = False
    Whether to display the game during training.
- path_to_save: str = 'qtable.csv'
    The path to save the Q-table.
- log_file: str = 'log.csv'
    The path to save the log file.


Deep-Q network based bot
------

In progress...
 

Credits
=======
 
Sprites: https://opengameart.org/content/bomb-party-the-complete-set

Game mechanics based on: https://github.com/Forestf90/Bomberman