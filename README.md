# AlphaPawn

## Overview

This repository contains a basic implementation of a weak neural network for playing chess, using Monte Carlo Tree Search (MCTS) and self-play to improve its skills. This network is designed to be a starting point for further development and training, with the ultimate goal of surpassing its current abilities.

### **Network Characteristics:**

*   Utilizes Monte Carlo Tree Search (MCTS) for decision-making
*   Employs self-play to improve its skills through reinforcement learning
*   Current strength: significantly weaker than Stockfish 30.11.2024

 Prerequisites
------------
*   Python 3
*   PyTorch
*   Chess engine library Python-Chess
*   MCTS implementation

## Repository Contents
-------------------

*   `mcts.py`: Implements the Monte Carlo Tree Search algorithm
*   `play_mtcs.py`: plays with the neural network

## Setup and Training

Install pytorch, py chess

Note: Training the network from scratch may take significant time and computational resources. You can consider using a [pre-trained](https://www.mediafire.com/file/b4wnzay27xpothy/mtcs.pth/file) model as a starting point **you can training it in mcts.py**.

## Future Development

This weak neural network is intended as a starting point for further development and enhancement. You can improve its strength by:

*   ~~By brute force, cash~~ prolonged training. **There were very few simulations. About 100 games.**
*   Integrating the network with more advanced chess-specific features (e.g., endgame tables, opening books)
*   Developing a more sophisticated MCTS implementation

## Feedback and Contributions
I encourage anyone interested in contributing to or providing feedback on this project to do so. You can create an issue or pull request in this repository, or simply send me a message with your thoughts and ideas.
