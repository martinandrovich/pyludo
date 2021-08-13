# Python LUDO game (pyludo)

A LUDO board-game simulator in Python3 with Q-learning players.

<p align="center">
  <img src="/assets/img/pyludo-screenshot.png">
</p>

<details>
<summary><strong>Game rules</strong></summary></br>

* Always four players.
* A player must roll a 6 to enter the board.
* Rolling a 6 does not grant a new dice roll.
* Globe positions are safe positions.
* The start position outside each home is considered a globe position
* A player token landing on a single opponent token sends the opponent token home if it is not on a globe position. If the opponent token is on a globe position the player token itself is sent home.
* A player token landing on two or more opponent tokens sends the player token itself home.
* If a player token lands on one or more opponent tokens when entering the board, all opponent tokens are sent home.
* A player landing on a star is moved to the next star or directly to goal if landing on the last star.
* A player in goal cannot be moved.

</details>

## Getting started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation

Installation of the project:

```bash
git clone https://github.com/martinandrovich/pyludo.git
cd pyludo
pip3 install -e .
```

## Usage

### Running example

You can visualize a an example match by running:

```bash
python3 pyludo/examples/game_rng_visualize.py
```

### State representation

The state is a numpy array of shape (4 players, 4 tokens)

The game state is represented as integers in a 4x4 numpy `state` array (4 players each with 4 tokens). Such that `state[i]` contains the `i`'th player's tokens, and `state[i][k]` will be the state of the `k`'th token of player `i`.

Home state is `-1`, and goal is `99` for all players. The common area range from `0-51` relative to player 0. The end lane is `52-56` for player 0, `57-61` for player 1 and so forth. A LudoPlayer is always fed a relative state in the `.play()` method, where the player itself is player 0. Note that when you move from home into the common area, you enter at position `1`, not `0`.

### Starting a game

Text.

### Creating additional players

Text.

## License

No license has been decided yet.

## Acknowledgments

- [Rasmus Haugaard](https://github.com/RasmusHaugaard) - original developer(s) of the pyludo project.
- [Haukur Kristinsson](https://github.com/haukri) - original developer(s) of the pyludo project.

