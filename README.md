# FSM planner

FSM is a learning-based planner that uses different sample generation improvement strategies to learn a heuristic function for classical planning. The heuristic function is a neural network trained on pairs of states and cost-to-goal estimates generated by regression from the goal state. The short time for sampling, training, and testing allows FSM to be competitive on the classical track of the IPC. The time dedicated to each of the mentioned steps is 15, 10, and 5 minutes, respectively.

FSM planner is completely based on the paper [Understanding Sample Generation Strategies for Learning Heuristic Functions in Classical Planning](https://arxiv.org/abs/2211.13316) and a fork from [Neural Fast Downward](https://github.com/PatrickFerber/NeuralFastDownward), which in turn derives from [Fast Downward](https://github.com/aibasel/downward).

## License

The following directory is not part of Fast Downward as covered by
this license:

- ./src/search/ext

For the rest, the following license applies:

```
Fast Downward is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

Fast Downward is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```