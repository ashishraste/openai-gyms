# OpenAI Gyms

This repository contains experiments on Reinforcement Learning using OpenAI Gyms.

## Environment Setup

1. Setup a Python 3 virtual environment

    ```bash
    python3 -m venv openai
    source openai/bin/activate
    ```

2. Install dependencies, including transitive dependencies, using `requirements.txt`

    ```bash
    pip install -r requirements.txt
    ```

## Cartpole Agent

The Cartpole problem is a classic control problem, also known as [Inverted Pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum),
where the centre of mass lies above the pivot point and the pole (or the pendulum) has to be balanced by moving the pivot under
the centre of mass around. Imagine humans trying to balance a stick on their palm.

[Cartpole-v0](https://github.com/openai/gym/wiki/CartPole-v0) OpenAI Gym provides us a simulation of this problem. This environment
is used in [Cartpole Agent](./cartpole-v0/README.md) to train a Q-learning agent that learns to balance the cartpole after few runs.
Read more about the agent [here](./cartpole-v0/README.md#Agent-Setup).
