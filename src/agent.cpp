#include "agent.h"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Agent::Agent(const std::string &netTop, double gamma, double epsilon,
             unsigned batchSize, unsigned targetSyncInterval,
             double learningRate, double momentum)
    : Agent(std::make_unique<Net>(netTop), gamma, epsilon, batchSize,
            targetSyncInterval, learningRate, momentum) {}

Agent::Agent(std::unique_ptr<Net> net, double gamma, double epsilon,
             unsigned batchSize, unsigned targetSyncInterval,
             double learningRate, double momentum)
    : net(std::move(net)), targetNet(std::make_unique<Net>(this->net.get())),
      gamma(gamma), epsilon(epsilon), learningRate(learningRate),
      momentum(momentum), batchSize(batchSize),
      targetSyncInterval(targetSyncInterval),
      outputSize(this->net->getTopology().back().neuronCount) {}

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

void Agent::addStep(const State &lastState, const State &newState,
                    const Action &action, double reward, bool done) {
  Transition t(lastState, action, reward, newState, done);
  trainShortMemory(t);
  remember(std::move(t));

  if (done) {
    n_games++;
    trainLongMemory();
  }
}

Action Agent::getAction(const State &state) {
  Action final_move(outputSize, 0);

  std::uniform_int_distribution<int> epsilonDist(0, 200);
  double curEps = currentEpsilon();

  if (epsilonDist(rng) < static_cast<int>(curEps)) {
    // Random exploration
    std::uniform_int_distribution<int> moveDist(0, outputSize - 1);
    final_move[moveDist(rng)] = 1;
  } else {
    // Exploitation: pick action with highest Q-value
    std::vector<double> qValues(outputSize);
    net->feedForward(state.data());
    net->getResults(qValues.data());

    int bestMove = 0;
    for (int i = 1; i < outputSize; i++) {
      if (qValues[i] > qValues[bestMove])
        bestMove = i;
    }
    final_move[bestMove] = 1;
  }

  return final_move;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

std::vector<double> Agent::computeTargets(const Transition &t) {
  std::vector<double> targets(outputSize);

  // Get current Q-values Q(s, ·) from policy network
  net->feedForward(t.state.data());
  net->getResults(targets.data());

  // Compute TD target for the taken action
  double Q_new = t.reward;
  if (!t.done) {
    std::vector<double> nextQ(outputSize);
    targetNet->feedForward(t.nextState.data());
    targetNet->getResults(nextQ.data());
    Q_new += gamma * *std::max_element(nextQ.begin(), nextQ.end());
  }

  // Update only the taken action's Q-value
  auto it = std::max_element(t.action.begin(), t.action.end());
  int actionIdx = static_cast<int>(std::distance(t.action.begin(), it));
  if (actionIdx >= 0 && actionIdx < outputSize)
    targets[actionIdx] = Q_new;

  // Re-feedforward original state for correct backProp gradients
  net->feedForward(t.state.data());

  return targets;
}

void Agent::trainOnTransition(const Transition &t, bool batchMode) {
  auto targets = computeTargets(t);
  net->backProp(targets.data(), learningRate, momentum, batchMode);

  totalTrainSteps++;
  if (totalTrainSteps % targetSyncInterval == 0)
    syncTargetNet();
}

void Agent::trainShortMemory(const Transition &t) {
  trainOnTransition(t, false);
}

void Agent::trainLongMemory() {
  if (memory.empty())
    return;

  // Build index array and shuffle for O(n) sampling without replacement
  std::vector<size_t> indices(memory.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);

  size_t count = std::min(static_cast<size_t>(batchSize), indices.size());
  for (size_t i = 0; i < count; ++i) {
    trainOnTransition(memory[indices[i]], true);
  }

  net->applyBatch();
}

// ---------------------------------------------------------------------------
// Memory & Sync
// ---------------------------------------------------------------------------

void Agent::remember(const Transition &t) {
  memory.push_back(t);
  while (memory.size() > kMaxMemorySize)
    memory.pop_front();
}

void Agent::syncTargetNet() {
  if (targetNet && net)
    targetNet->createCopyFrom(net.get());
}
