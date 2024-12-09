# RL-Planes
**Authors**: Michal Bělohlávek a Tomáš Procházka

**We advise to download the code and run this project locally to enjoy the animation at the end. Moreover, one can pleasantly move within the configuraion file.**

## About
This project concerns itself with the application of Multi-Agent Reinforcement Learning for automatic routing of planes. We build on the implementation of authors [] and adapt and modify their code to suit our needs. The architecture of this beast comprises of Graph Convolutional Networks, Recurrent Neural Networks and other beautiful machine learning structures. These components are combined to create outputs that almost look like magic, all while employing an extremely thought-out process. 

## How to run the project
Description on how to run the project can be found ...

## Our contributions
Our contributions mainly consist of the following
    - adaptation to traffic air control for efficient routing of planes
    - stateless version of the NetMon class -> the training can be split across multiple gpus
    - generalization to variable neighbourhoods
    - addition of prioritized replay buffer
    - addition of regularization as we deal with higher dimensional data, consequently using more layers within our models

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


  
