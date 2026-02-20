# GenNet

A lightweight, **Qt-free** C++17 neural network library for building, training, and evolving feedforward neural networks. Designed for real-time applications like game AI.

## Features

- **Flexible Topology** — Define networks via string notation with per-layer activation and aggregation functions
- **Backpropagation** — Gradient-based training with configurable learning rate (η) and momentum (α)
- **Batch Learning** — Accumulate gradients before applying weight updates
- **Genetic Evolution** — Mutate and evolve populations with elitism or simulated annealing
- **DQN Agent** — Deep Q-Network with experience replay and target network
- **Multi-threaded** — Parallel mutation for large populations
- **Save / Load** — Serialize networks to CSV
- **No Dependencies** — Pure C++17 standard library

## Architecture

```
GenNet/
├── src/
│   ├── fastrandom.h       # xorshift128+ PRNG (fast, thread-safe)
│   ├── neuron.h/cpp       # Neuron: activation, aggregation, mutation
│   ├── net.h/cpp          # Net: feedforward, backprop, save/load
│   ├── population.h/cpp   # Population: genetic evolution
│   └── agent.h/cpp        # DQN reinforcement learning agent
└── docs/
    ├── Net.md             # Detailed Net documentation
    ├── Population.md      # Evolution strategies & API
    └── Agent.md           # DQN architecture & API
```

## Quick Start

### Create & Train a Network

```cpp
#include "net.h"

Net net("4_INPUT,8_SUM_RELU,2_SUM_IDENTITY");

double input[4]  = {1.0, 0.5, -0.3, 0.8};
double target[2] = {1.0, 0.0};

// Forward pass
net.feedForward(input);
double output[2];
net.getResults(output);

// Train (learning rate=0.15, momentum=0.05)
net.backProp(target, 0.15, 0.05);
```

### Topology Format

Networks are defined as comma-separated layers: `<count>_<aggregation>_<activation>`

```
24_INPUT,16_SUM_RELU,4_SUM_TANH
   │         │   │        │   └─ activation
   │         │   │        └─── aggregation
   │         │   └──────────── neuron count
   │         └──────────────── hidden layer
   └────────────────────────── input layer (shorthand)
```

**Aggregation:** `SUM`, `AVG`, `MAX`, `MIN`, `INPUT`
**Activation:** `TANH`, `RELU`, `LEAKYRELU`, `SIGMOID`, `SMAX`, `SOFTPLUS`, `IDENTITY`, `NONE`

→ See [docs/Net.md](docs/Net.md) for full API reference and file format.

### Genetic Evolution

```cpp
#include "population.h"

Population pop("24_INPUT,16_SUM_RELU,4_SUM_TANH", 100);

// Evaluate fitness
int* scores = pop.scoreMap();
for (unsigned i = 0; i < pop.getSize(); i++)
    scores[i] = evaluate(pop.netAt(i));

// Evolve
pop.evolve(bestIdx, 0.1, 0.2);

// Or with simulated annealing
pop.evolveWithSimulatedAnnealing(0.1, 0.2, 0.95);
```

→ See [docs/Population.md](docs/Population.md) for evolution strategies and multi-threading.

### DQN Reinforcement Learning

```cpp
#include "agent.h"

Agent agent("11_INPUT,256_SUM_RELU,3_SUM_IDENTITY");

State state = getState();
Action action = agent.getAction(state);
auto [nextState, reward, done] = env.step(action);
agent.addStep(state, nextState, action, reward, done);
```

→ See [docs/Agent.md](docs/Agent.md) for DQN architecture, hyperparameters, and feature status.

## Build

GenNet builds as a **static library** using qmake (no Qt dependency):

```bash
qmake GenNet.pro
make
```

Output: `libGenNet.a`

### Integration

Add to your `.pro` file:

```qmake
LIBS += -L$$PWD/../GenNet/release -lGenNet
INCLUDEPATH += $$PWD/../GenNet/src
DEPENDPATH += $$PWD/../GenNet/src
PRE_TARGETDEPS += $$PWD/../GenNet/release/libGenNet.a
```

## License

All rights reserved.
