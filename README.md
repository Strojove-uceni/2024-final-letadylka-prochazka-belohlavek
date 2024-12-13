# RL-Planes
**Authors**: Michal Bělohlávek a Tomáš Procházka

**We advise to download the code and run this project locally to enjoy the animation at the end. Moreover, one can pleasantly move within the configuraion file.**

## About
This project concerns itself with the application of Multi-Agent Reinforcement Learning for automatic routing of planes. We build on the implementation of authors (view References) and adapted and modified their code to suit our needs. The architecture of this beast comprises of Graph Convolutional Networks, Recurrent Neural Networks and other beautiful machine learning structures. These components are combined to create outputs that almost look like magic, all while employing an extremely thought-out process. 

## How to Run the Project
Quick demonstration is included in demo.ipynb.

## Our Contributions
Our contributions mainly consist of the following:
    - adaptation of multia agent reinforcement learning to air traffic control for efficient routing
    - generalization to variable number of edges
    - addition of prioritized sampling from the replay buffer
    - addition of regularization as we deal with higher dimensional data, consequently using more layers within our models
    - augmented the structure to support dynamic environment
    - state-less version of the NetMon class -> the training can be split across multiple gpus

## Available Models to Try Out
### Q-value Prediction
  - DGN = Graph Convolutional Reinforcement Learning
  - DQN = Deep Q-Networks
  - Comm_net = Communication Network
### State Aggregation
  - GCN = Graph Convolutional Network
  - SUM = simple sum
Further description of these methods can be found within description file.

## Multi-GPU support
We have provided code for multi-GPU support for the NetMon class created by the authors. We have used simple DataParallel python structure for a split across mulitple GPUs within a single compute node. Additionally, we have modified the code to support distribution of the Q-value predictor and the NetMon class between 2 GPUs, therefore allowing for slightly bigger models. 

## Results
We were able to train multiple models that effectively control ten planes on a relatively sparse graph. The models have relatively stable performance with numerous planes reaching the target on their own. We also implemented an action mask, that prevents taking unwanted actions like flying back or returning to the already visited point, yet the agents have a valid choice in the overwhleming majority of instances. Our agents are not proned to exploit the reward system, despite it being quite a dense one. We performed a sweep of the hyperparameter space, which significantly enhanced the model performance. On this GitHub repo, we provide a comprehensive overview of our work, code and results in the demo file.

### References
  - Graph MARL original implementation GitHub repo: https://github.com/jw3il/graph-marl/tree/main?tab=readme-ov-file, Weil, J., Bao, Z., Abboud, O., & Meuser, T. (2024). Towards Generalizability of Multi-Agent Reinforcement  Learning in Graphs with Recurrent Message Passing [Conference paper]. Proceedings of the 23rd International Conference on  Autonomous Agents and Multiagent Systems
  - Source for the Replay Buffer for experience based prioritization: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py

  
