# general-OpenAI-gym-agent
General OpenAI gym agent trained using Deep Q Network learning (DQN) algorithm 

## Introduction
The goal of this project is to train an agent for the environments with discrete action space and continuous observation space from OpenAI gym.

## Algorithm Overview
Deep Q Network learning (DQN) algorithm is implement with neural network from Keras library. Neural network is used for funciton approximation purpose and trained by past observations with Experience Replay technique using Bellman equation by back-propagation. Past experients are stored and randomly sampled for training at each step. The agent performs pure exploration at beginning followed by ε-greedy policy.

## Requirements
* python 2.7
* numpy
* OpenAI's gym
* Keras with tensorflow
* h5py

## Result 
### LunarLander-v2

![LunarLander-v2](https://github.com/vivianhylee/general-OpenAI-gym-agent/blob/master/trained%20agent/LunarLander-v2_video.gif)

