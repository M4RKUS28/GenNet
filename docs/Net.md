# Net — Feedforward Neural Network

Core neural network class providing feedforward propagation, backpropagation, mutation, and serialization.

## Architecture

Each `Net` consists of layers of `Neuron` objects. Every layer (except the output) includes an additional bias neuron with a fixed output of `1.0`.

```
Input Layer → [Hidden Layers...] → Output Layer
   └── Each neuron connects to all neurons in the next layer
   └── Bias neuron added per layer (output = 1.0)
```

## Topology Format

```
<neuronCount>_<aggregation>_<activation>
```

Input layer shorthand: `<neuronCount>_INPUT`

**Example:** `24_INPUT,16_SUM_RELU,4_SUM_TANH`

### Aggregation Functions

| Name    | Formula                      |
| ------- | ---------------------------- |
| `SUM`   | Σ (weight × input)           |
| `AVG`   | Σ (weight × input) / n       |
| `MAX`   | max(weight × input)          |
| `MIN`   | min(weight × input)          |
| `INPUT` | Input layer (no aggregation) |

### Activation Functions

| Name        | f(x)              | f'(x)           |
| ----------- | ----------------- | --------------- |
| `TANH`      | tanh(x)           | 1 − x²          |
| `RELU`      | max(0, x)         | x > 0 ? 1 : 0   |
| `LEAKYRELU` | max(x, 0.1x)      | x > 0 ? 1 : 0.1 |
| `SIGMOID`   | 1 / (1 + e⁻ˣ)     | σ(x)(1 − σ(x))  |
| `SMAX`      | eˣ / Σeˣ (stable) | s(1 − s)        |
| `SOFTPLUS`  | ln(1 + eˣ)        | σ(x)            |
| `IDENTITY`  | x                 | 1               |
| `NONE`      | x                 | 1               |

## Usage

### Create

```cpp
#include "net.h"

// From topology string
Net net("24_INPUT,16_SUM_RELU,4_SUM_TANH");

// From file
bool ok;
Net loaded("model.csv", ok);

// Copy from existing
Net copy(&original);
```

### Feedforward & Backprop

```cpp
double input[24] = { /* ... */ };
double target[4] = { /* ... */ };

net.feedForward(input);

double output[4];
net.getResults(output);

// Train (eta=0.15, alpha=0.05)
net.backProp(target, 0.15, 0.05);
```

### Batch Learning

```cpp
Net net("3_INPUT,5_SUM_RELU,2_SUM_SIGMOID", 0.01, true);

for (auto& sample : dataset) {
    net.feedForward(sample.input);
    net.backProp(sample.target, 0.15, 0.05, true);  // accumulate
}
net.applyBatch();  // apply all at once
```

### Save & Load

```cpp
net.saveTo("model.csv");
net.loadFrom("model.csv");
```

### Mutation (Genetic Algorithms)

```cpp
net.mutate(0.1, 0.2);  // rate=10%, range=0.2
```

### Comparing Networks

```cpp
double diff = netA.getDifferenceFrom(&netB);  // average weight difference
```

## API Reference

| Method                                 | Description                        |
| -------------------------------------- | ---------------------------------- |
| `feedForward(input)`                   | Run forward pass                   |
| `getResults(output)`                   | Copy output neuron values          |
| `backProp(targets, eta, alpha, batch)` | Gradient descent step              |
| `applyBatch()`                         | Apply accumulated batch gradients  |
| `mutate(rate, range)`                  | Gaussian mutation with clipping    |
| `createCopyFrom(other)`                | Copy weights from another net      |
| `getDifferenceFrom(other)`             | Average absolute weight difference |
| `saveTo(path)`                         | Save to CSV file                   |
| `loadFrom(path)`                       | Load from CSV file                 |
| `getTopology()`                        | Get layer configuration (const&)   |
| `getTopologyStr()`                     | Get topology as string             |
| `getConnectionWeight(l, nFrom, nTo)`   | Get specific weight                |
| `getNeuronValue(l, n)`                 | Get neuron output                  |
| `recentAverageError()`                 | Smoothed training error            |

## File Format

```
24_INPUT,16_SUM_RELU,4_SUM_TANH   ← topology string
------
0.123,0.456,...                     ← neuron 0, layer 0 weights
-0.789,0.012,...                    ← neuron 1, layer 0 weights
...                                 ← (including bias neuron)
                                    ← blank line between layers
0.345,-0.678,...                    ← neuron 0, layer 1 weights
...
```
