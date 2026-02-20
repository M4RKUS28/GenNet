#ifndef AGENT_H
#define AGENT_H

#include "net.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using State = std::vector<double>;
using Action = std::vector<int>;

/**
 * @brief A single experience tuple (s, a, r, s', done) stored in replay memory.
 */
struct Transition {
  State state;
  Action action;
  double reward = 0.0;
  State nextState;
  bool done = false;

  Transition() = default;
  Transition(State state, Action action, double reward, State nextState,
             bool done)
      : state(std::move(state)), action(std::move(action)), reward(reward),
        nextState(std::move(nextState)), done(done) {}
};

/**
 * @brief DQN reinforcement learning agent with experience replay and target
 * network.
 *
 * Uses an epsilon-greedy policy with linear decay. The target network is
 * periodically synced from the policy network every `targetSyncInterval`
 * training steps for stable Q-value estimation.
 */
class Agent {
public:
  /**
   * @brief Construct agent from a topology string.
   * @param netTop  Topology string (e.g. "4_INPUT,64_SUM_RELU,2_SUM_IDENTITY")
   * @param gamma   Discount factor for future rewards (0..1)
   * @param epsilon Initial epsilon for exploration (decays with n_games)
   * @param batchSize Number of samples per long-memory training batch
   * @param targetSyncInterval Sync target net every N training steps
   * @param learningRate Learning rate (eta) for backpropagation
   * @param momentum Momentum (alpha) for backpropagation
   */
  Agent(const std::string &netTop, double gamma = 0.9, double epsilon = 80.0,
        unsigned batchSize = 1000, unsigned targetSyncInterval = 100,
        double learningRate = 0.01, double momentum = 0.15);

  /**
   * @brief Construct agent from an existing network (takes ownership).
   */
  Agent(std::unique_ptr<Net> net, double gamma = 0.9, double epsilon = 80.0,
        unsigned batchSize = 1000, unsigned targetSyncInterval = 100,
        double learningRate = 0.01, double momentum = 0.15);

  ~Agent() = default;

  // Non-copyable, movable
  Agent(const Agent &) = delete;
  Agent &operator=(const Agent &) = delete;
  Agent(Agent &&) = default;
  Agent &operator=(Agent &&) = default;

  // --- Core API ---

  /**
   * @brief Perform one step: train on transition, remember it, and if the
   * episode is done, increment n_games and run long-memory training.
   */
  void addStep(const State &lastState, const State &newState,
               const Action &action, double reward, bool done);

  /**
   * @brief Select an action using epsilon-greedy policy.
   */
  Action getAction(const State &state);

  // --- Accessors ---

  unsigned getNumGames() const { return n_games; }
  double getEpsilon() const { return currentEpsilon(); }
  double getRecentError() const { return net->recentAverageError(); }
  const Net *getNet() const { return net.get(); }
  const Net *getTargetNet() const { return targetNet.get(); }

private:
  void trainOnTransition(const Transition &t, bool batchMode);
  void trainShortMemory(const Transition &t);
  void trainLongMemory();
  void remember(const Transition &t);

  /**
   * @brief Compute target Q-values for a transition.
   *
   * Uses target network for next-state value estimation (Double DQN style).
   * Re-runs feedForward on the original state afterwards so backProp
   * computes correct gradients.
   */
  std::vector<double> computeTargets(const Transition &t);

  /**
   * @brief Copy weights from policy network to target network.
   */
  void syncTargetNet();

  double currentEpsilon() const {
    double e = epsilon - static_cast<double>(n_games);
    return e > 0.0 ? e : 0.0;
  }

  // --- Networks ---
  std::unique_ptr<Net> net;
  std::unique_ptr<Net> targetNet;

  // --- Hyperparameters ---
  double gamma;
  double epsilon;
  double learningRate;
  double momentum;
  unsigned batchSize;
  unsigned targetSyncInterval;

  // --- State ---
  std::deque<Transition> memory;
  static constexpr unsigned kMaxMemorySize = 100000;
  unsigned n_games = 0;
  unsigned totalTrainSteps = 0;
  int outputSize = 0;

  // --- RNG ---
  std::mt19937 rng{std::random_device{}()};
};

#endif // AGENT_H
