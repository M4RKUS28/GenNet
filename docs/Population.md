# Population — Genetic Evolution

Manages a collection of neural networks and evolves them using genetic algorithms with optional multi-threaded mutation.

## Evolution Strategies

### 1. Basic Evolution (`evolve`)

Copies the best network to the rest of the population with varying mutation levels:

```
Population split:
├── 75%  — copy from best, standard mutation (user-defined rate/range)
├── 12.5% — copy from best, low mutation (rate=0.01, range=0.2)
└── 12.5% — copy from best, high mutation (rate=0.03, range=0.3)
```

The best network is **never** overwritten (elitism).

### 2. Simulated Annealing (`evolveWithSimulatedAnnealing`)

Temperature-controlled evolution that probabilistically accepts worse solutions:

```
For each network i ≠ best:
  if score[i] >= score[best]  → keep & mutate in-place
  else                        → accept with P = exp((score_i/score_best - 1) / T)
                                 reject → overwrite with best & mutate

Temperature decays:  T = T × decay_rate
Mutation rate scales: effective_rate = base_rate × T
```

As temperature decreases, the algorithm becomes increasingly greedy — accepting fewer worse solutions and applying smaller mutations.

## Usage

```cpp
#include "population.h"

// Create 100 networks
Population pop("24_INPUT,16_SUM_RELU,4_SUM_TANH", 100);

// Game loop — evaluate each network
int* scores = pop.scoreMap();
for (unsigned i = 0; i < pop.getSize(); i++) {
    Net* net = pop.netAt(i);
    scores[i] = runGame(net);
}

// Option A: basic evolution
unsigned bestIdx = /* find best score index */;
pop.evolve(bestIdx, 0.1, 0.2);

// Option B: simulated annealing (auto-finds best)
pop.evolveWithSimulatedAnnealing(0.1, 0.2, 0.95);
```

### Multi-threaded Mutation

```cpp
// Enable threads in constructor (last parameter)
Population pop("24_INPUT,16_SUM_RELU,4_SUM_TANH", 100, 0.1, 1.0, true);
//                                                                 ^^^^
```

The population automatically distributes mutation work across `hardware_concurrency - 1` threads.

## Constructor Parameters

| Parameter            | Default | Description                                |
| -------------------- | ------- | ------------------------------------------ |
| `topology`           | —       | Network topology string                    |
| `size`               | —       | Number of networks in population           |
| `initRange`          | 0.1     | Initial random weight range                |
| `initTemperature`    | 1.0     | Starting temperature (simulated annealing) |
| `useMutationThreads` | false   | Enable multi-threaded mutation             |

## API Reference

| Method                                             | Description                      |
| -------------------------------------------------- | -------------------------------- |
| `evolve(bestIdx, rate, range)`                     | Basic elitism-based evolution    |
| `evolveWithSimulatedAnnealing(rate, range, decay)` | Temperature-controlled evolution |
| `netAt(index)`                                     | Access network at index          |
| `scoreMap()`                                       | Get writable score array         |
| `getSize()`                                        | Population size                  |
| `getEvolutionCount()`                              | Number of completed evolutions   |
| `getTemperature()`                                 | Current annealing temperature    |
| `setTemperature(t)`                                | Override temperature             |
