# Python LUDO game (pyludo)

A python3 LUDO simulator.

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

You can visualize a ... by running

```bash
python3 -m pyludo.examples.visualizeRandomPlayerMatch
```

### Starting a game

Text.

### State representation

The state is a numpy array of shape (4 players, 4 tokens)

`state[i]` will then be the i'th player's tokens, and `state[i][k]` will be the value of the k'th token of player i.

Home is -1, and goal is 99 for all players.
The common area is from 0 to 51 but relative to player 0.
The end lane is 52 to 56 for player 0, 57 to 61 for player 1, etc.

A LudoPlayer is always fed a relative state, where the player itself is player 0.

Note that when you move from home into the common area, you enter at position 1, not 0.

### Creating additional players

Text.

## License

No license has been decided yet.

## Acknowledgments

- [Martin Androvich][androvich-git] - original developer(s) of the pyludo project.
- [Martin Androvich][androvich-git] - original developer(s) of the pyludo project.

