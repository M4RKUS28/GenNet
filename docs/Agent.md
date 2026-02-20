# DQN Agent

A Deep Q-Network (DQN) reinforcement learning agent with experience replay and target network stabilization.

## Architecture

```
State → [Policy Network] → Q(s, a) → ε-greedy → Action
                ↑ sync every N steps
         [Target Network] → stable Q-targets for TD learning
```

### Training Flow

1. **`addStep()`** — Called every environment step:
   - Computes TD target and trains immediately (short memory)
   - Stores transition in replay buffer
   - On episode end (`done=true`): samples batch from replay buffer (long memory)

2. **`computeTargets()`** — Bellman equation:

   ```
   Q_target(a) = reward + γ · max_a' Q_targetNet(s', a')
   ```

   Only the taken action's Q-value is updated; all others remain unchanged (zero gradient).

3. **`getAction()`** — ε-greedy policy:
   ```
   ε = max(0, initial_ε − n_games)
   P(random) = ε / 201
   ```

## Usage

```cpp
#include "agent.h"

// Create agent with topology, gamma=0.99, epsilon=80, batchSize=1000
Agent agent("11_INPUT,256_SUM_RELU,3_SUM_IDENTITY", 0.99, 80.0, 1000);

// Game loop
State state = getState();
Action action = agent.getAction(state);
auto [nextState, reward, done] = env.step(action);
agent.addStep(state, nextState, action, reward, done);
```

### Constructor Parameters

| Parameter            | Default | Description                                               |
| -------------------- | ------- | --------------------------------------------------------- |
| `gamma`              | 0.9     | Discount factor for future rewards                        |
| `epsilon`            | 80.0    | Initial exploration rate (decays linearly with `n_games`) |
| `batchSize`          | 1000    | Samples per long-memory training batch                    |
| `targetSyncInterval` | 100     | Sync target net every N training steps                    |
| `learningRate`       | 0.01    | Learning rate (η) for backpropagation                     |
| `momentum`           | 0.15    | Momentum (α) for backpropagation                          |

## Feature Status

| Feature           | Status         | Description                             |
| ----------------- | -------------- | --------------------------------------- |
| Bellman TD Target | ✅ Implemented | `Q = r + γ · max Q_target(s')`          |
| Target Network    | ✅ Implemented | Periodic hard sync from policy net      |
| Experience Replay | ✅ Implemented | Deque buffer, max 100k transitions      |
| Epsilon Decay     | ✅ Implemented | Linear: `ε = initial_ε − n_games`       |
| Reward Clipping   | ❌ Missing     | Helps training stability                |
| Gradient Clipping | ❌ Missing     | Prevents exploding gradients            |
| Double DQN        | ❌ Missing     | Policy selects action, target evaluates |
| Per-step Replay   | ❌ Missing     | Currently only replays at episode end   |

## API Reference

### Public Methods

| Method                                            | Description                           |
| ------------------------------------------------- | ------------------------------------- |
| `addStep(state, nextState, action, reward, done)` | Process one environment step          |
| `getAction(state)`                                | Get ε-greedy action (one-hot encoded) |
| `getNumGames()`                                   | Number of completed episodes          |
| `getEpsilon()`                                    | Current exploration rate              |
| `getRecentError()`                                | Recent average training error         |
| `getNet()`                                        | Access policy network (const)         |
| `getTargetNet()`                                  | Access target network (const)         |
