#ifndef AGENT_H
#define AGENT_H

#include "net.h"
#include <algorithm>
#include <deque>
#include <vector>

typedef std::vector<double> State;
typedef std::vector<int> Action;

class Agent {
public:
  Agent(std::string netTop, double gamma = 0.9, double epsilon = 80.0,
        unsigned batchSize = 1000, unsigned targetSyncInterval = 100);
  Agent(Net *net, double gamma = 0.9, double epsilon = 80.0,
        unsigned batchSize = 1000, unsigned targetSyncInterval = 100);
  ~Agent();

  struct MEM_ENTRY {
    MEM_ENTRY() {}
    MEM_ENTRY(State state, Action action, double reward, State nextState,
              bool done)
        : state(state), action(action), reward(reward), nextState(nextState),
          done(done) {}

    State state;
    Action action;
    double reward;
    State nextState;
    bool done;
  };

  void train_short_memory(const MEM_ENTRY &mem);
  void train_long_memory();
  void remember(const MEM_ENTRY &mem);

  /**
   * @brief Computes the DQN target values for a given memory entry.
   *
   * Uses the target network (if available) for stable Q-value estimation.
   * Re-runs feedForward on the original state so backProp computes
   * correct gradients.
   */
  double *targetValue(const MEM_ENTRY &mem);

  /**
   * @brief Copies weights from the policy network to the target network.
   * Called automatically every `targetSyncInterval` training steps.
   */
  void syncTargetNet();

public:
  void addStep(State lastState, State newState, Action action, double reward,
               bool done);
  Action getAction(const State &state);

  std::deque<MEM_ENTRY> memory;
  Net *net = nullptr;       // Policy network (owned)
  Net *targetNet = nullptr; // Target network for stable Q-targets (owned)
  const unsigned maxMemSize = 100000;
  unsigned batchSize;
  double gamma;
  double epsilon;
  unsigned n_games = 0;              // Game counter for epsilon decay
  unsigned targetSyncInterval = 100; // Sync target net every N training steps
  unsigned totalTrainSteps = 0;      // Total backprop calls for sync timing
};

#endif // AGENT_H
