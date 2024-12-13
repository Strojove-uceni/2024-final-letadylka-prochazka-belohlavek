# RL-Planes
**Authors**: Michal Bělohlávek a Tomáš Procházka

**We advise to download the code and run this project locally to enjoy the animation at the end. Moreover, one can pleasantly move within the configuraion file.**

## About
This project concerns itself with the application of Multi-Agent Reinforcement Learning for automatic routing of planes. We build on the implementation of authors in [previous works](https://github.com/jw3il/graph-marl) and adapted and modified their code to suit our needs. The architecture of this beast comprises of Graph Convolutional Networks, Recurrent Neural Networks and other beautiful machine learning structures. These components are combined to create outputs that almost look like magic, all while employing an extremely thought-out process. 

## How to Run the Project
Quick demonstration is included in demo.ipynb.

## Our Contributions
Our contributions mainly consist of the following:
- adaptation of multia agent reinforcement learning to air traffic control for efficient routing
- generalization to variable number of edges
- addition of prioritized sampling from the replay buffer
- addition of regularization as we deal with higher dimensional data, consequently using more layers within our models
- augmented the structure to support dynamic environment
- state-less version of the NetMon class $\rightarrow$ the training can be split across multiple gpus

## Available Models to Try Out
### Q-value Prediction
- DGN = Graph Convolutional Reinforcement Learning
- DQN = Deep Q-Network
- Comm_net = Communication Network
### State Aggregation
- GCN = Graph Convolutional Network
- SUM = simple sum
Further description of these methods can be found within description file.

## Multi-GPU support
We have provided code for multi-GPU support for the NetMon class created by the [authors](https://github.com/jw3il/graph-marl). We have used simple DataParallel PyTorch structure for a split across mulitple GPUs within a single compute node. Additionally, we have modified the code to support distribution of the Q-value predictor and the NetMon class between 2 GPUs, therefore allowing for slightly bigger models. 

## Results
We successfully trained multiple models capable of effectively controlling ten planes on a relatively sparse graph. The models demonstrated stable performance, with many planes autonomously reaching their targets. Additionally, we implemented an action mask to prevent undesired actions, such as flying backward or returning to previously visited points, while ensuring that agents retain valid choices in the vast majority of scenarios.

Our agents do not exploit the reward system, even though it is relatively dense. A thorough sweep of the hyperparameter space significantly enhanced the models' performance. A comprehensive overview of our work, including the code and results, is available in the demo file on this GitHub repository.

### References
  - Graph MARL original implementation GitHub repo: https://github.com/jw3il/graph-marl/tree/main?tab=readme-ov-file, Weil, J., Bao, Z., Abboud, O., & Meuser, T. (2024). Towards Generalizability of Multi-Agent Reinforcement  Learning in Graphs with Recurrent Message Passing [Conference paper]. Proceedings of the 23rd International Conference on  Autonomous Agents and Multiagent Systems
  - Source for the Replay Buffer for experience based prioritization: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py

  
