#include "agent.h"

Agent::Agent(std::string netTop, double gamma, double epsilon,
             unsigned int batchSize, unsigned targetSyncInterval)
    : Agent(new Net(netTop), gamma, epsilon, batchSize, targetSyncInterval) {}

Agent::Agent(Net *net, double gamma, double epsilon, unsigned int batchSize,
             unsigned targetSyncInterval)
    : net(net), targetNet(new Net(net)), batchSize(batchSize), gamma(gamma),
      epsilon(epsilon), targetSyncInterval(targetSyncInterval) {}

Agent::~Agent() {
  delete net;
  delete targetNet;
}

void Agent::remember(const MEM_ENTRY &mem) {
  memory.push_back(mem);
  while (memory.size() > maxMemSize)
    memory.pop_front();
}

static int argmax(const Action &vec) {
  if (vec.empty())
    return -1;
  return static_cast<int>(
      std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
}

double *Agent::targetValue(const MEM_ENTRY &mem) {
  int size = net->getTopology().back().neuronCount;
  double *retVal = new double[size];

  // Step 1: Get current Q-values for the state
  net->feedForward(mem.state.data());
  net->getResults(retVal);

  // Step 2: Compute Q_new from target network (stable targets)
  double Q_new = mem.reward;
  if (!mem.done) {
    double *predic = new double[size];

    // Use TARGET network for next-state Q-value estimation
    targetNet->feedForward(mem.nextState.data());
    targetNet->getResults(predic);

    double max = *std::max_element(predic, predic + size);
    Q_new = mem.reward + gamma * max;
    delete[] predic;
  }

  // Step 3: Set the target for the taken action
  int actionIdx = argmax(mem.action);
  if (actionIdx >= 0 && actionIdx < size)
    retVal[actionIdx] = Q_new;

  // Step 4: Re-feedforward the ORIGINAL state so that backProp
  //         computes gradients for the correct activations
  net->feedForward(mem.state.data());

  return retVal;
}

void Agent::addStep(State lastState, State newState, Action action,
                    double reward, bool done) {
  MEM_ENTRY memEntr(lastState, action, reward, newState, done);
  train_short_memory(memEntr);
  remember(memEntr);

  if (done) {
    n_games++;
    train_long_memory();
  }
}

Action Agent::getAction(const State &state) {
  int size = net->getTopology().back().neuronCount;
  Action final_move(size, 0);

  // Epsilon-greedy with decay: epsilon decreases as n_games grows
  double current_epsilon = epsilon - static_cast<double>(n_games);
  if (current_epsilon < 0.0)
    current_epsilon = 0.0;

  if (std::rand() % 201 < static_cast<int>(current_epsilon)) {
    // Random exploration
    int move = std::rand() % size;
    final_move[move] = 1;
  } else {
    // Exploitation: pick best action from policy network
    double *rawData = new double[size];
    net->feedForward(state.data());
    net->getResults(rawData);

    int move = 0;
    for (int i = 1; i < size; i++) {
      if (rawData[i] > rawData[move])
        move = i;
    }
    delete[] rawData;
    final_move[move] = 1;
  }

  return final_move;
}

void Agent::train_short_memory(const MEM_ENTRY &mem) {
  double *targetVal = targetValue(mem);
  net->backProp(targetVal, 0.01, 0.15, false);
  delete[] targetVal;

  totalTrainSteps++;
  if (totalTrainSteps % targetSyncInterval == 0)
    syncTargetNet();
}

void Agent::train_long_memory() {
  std::deque<MEM_ENTRY> sampleDeque = memory;

  std::random_device rd;
  std::mt19937 gen(rd());

  for (std::size_t i = 0; i < batchSize && !sampleDeque.empty(); ++i) {
    std::size_t randomIndex = std::uniform_int_distribution<std::size_t>(
        0, sampleDeque.size() - 1)(gen);

    double *targetVal = targetValue(sampleDeque[randomIndex]);
    net->backProp(targetVal, 0.01, 0.15, true);
    delete[] targetVal;

    sampleDeque.erase(sampleDeque.begin() + randomIndex);
  }

  net->applyBatch();

  totalTrainSteps++;
  if (totalTrainSteps % targetSyncInterval == 0)
    syncTargetNet();
}

void Agent::syncTargetNet() {
  if (targetNet && net)
    targetNet->createCopyFrom(net);
}
