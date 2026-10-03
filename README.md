# micrograd (from scratch)

A minimal automatic differentiation engine and neural network library, written
from scratch in Python while following Andrej Karpathy's
[Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0)
micrograd lecture. Original project: [karpathy/micrograd](https://github.com/karpathy/micrograd).

## What it does

- **`Value` class**: wraps a single number and records the operations that
  produce it (`+`, `-`, `*`, `/`, `**`, `exp`, `tanh`), building a
  computation graph as you go.
- **Backpropagation**: `.backward()` sorts the graph topologically and applies
  the chain rule in reverse, filling in the gradient of the output with respect
  to every value in the graph.
- **Neural network**: `Neuron`, `Layer` and `MLP` classes built on top of
  `Value`, trained with mean squared error and gradient descent on a small
  example dataset.
- **Visualization**: `draw_dot()` renders the computation graph with each
  node's data and gradient using Graphviz.
