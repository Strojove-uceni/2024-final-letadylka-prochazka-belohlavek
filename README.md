# RL-Planes
**Authors**: Michal Bělohlávek a Tomáš Procházka

## About
This project concerns itself with the application of Multi-Agent Reinforcement Learning for automatic routing of planes. We build on the implementation of authors [] and adapt and modify their code to suit our needs. Architecture of this best consists of Graph Convolutional Networks, Reccurent Neural Networks and other beatiful machine learning structures that create this routing magic.

## How to run the project
Description on how to run the project can be found ...

## Our contributions
Our contributions mainly consist of the following
    - adaptation to traffic air control for efficient routing of planes
    - stateless version of the NetMon class -> the training can be split across multiple gpus
    - generalization to variable neighbourhoods
    - addition of prioritized replay buffer

## Available models to try out
### Q-value prediction
  - DGN = Graph Convolutional Reinforcement Learning
  - DQN = Deep Q-Networks
  - Comm_net = Communication Network
### State aggregation
  - GCN = Graph Convolutional Network
  - SUM = simple sum
Further description of these methods can be found within description file.

### References


  
