#ifndef POPULATION_H
#define POPULATION_H

#include "net.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

/**
 * @brief Manages a population of neural networks for genetic evolution.
 *
 * Supports both basic evolution (copy-from-best + mutate) and simulated
 * annealing evolution with probabilistic acceptance of worse solutions.
 * Optionally parallelizes mutation across multiple threads.
 */
class Population {
public:
  Population(const std::string &topology, unsigned size, double initRange = 0.1,
             double initTemperature = 1.0, bool useMutationThreads = false);
  ~Population();

  // Non-copyable, non-movable (owns net array)
  Population(const Population &) = delete;
  Population &operator=(const Population &) = delete;

  /**
   * @brief Evolve population by copying from the best network and mutating.
   * @param bestIdx Index of the best-performing network
   * @param mutationRate Mutation rate
   * @param mutationRange Range of Gaussian mutation noise
   */
  void evolve(unsigned bestIdx, double mutationRate,
              double mutationRange = 0.2);

  /**
   * @brief Evolve using simulated annealing: worse solutions accepted with
   *        probability exp((score_i/score_best - 1) / temperature).
   */
  void evolveWithSimulatedAnnealing(double mutationRate,
                                    double mutationRange = 0.2,
                                    double temperatureDecay = 0.9);

  // --- Accessors ---
  Net *netAt(unsigned index);
  const Net *netAt(unsigned index) const;
  int *scoreMap() { return scores.get(); }
  const int *scoreMap() const { return scores.get(); }
  unsigned getEvolutionCount() const { return evolutionCount; }
  unsigned getSize() const { return populationSize; }
  double getTemperature() const { return temperature; }
  void setTemperature(double t) { temperature = t; }

private:
  struct MutationTask {
    int sourceIdx;
    int targetIdx;
  };

  void dispatchMutation(
      std::vector<std::pair<std::thread, std::atomic<bool> *>> &threads,
      const std::vector<MutationTask> &tasks, unsigned maxThreads,
      double mutRate, double mutRange);

  static void executeMutation(Net **nets,
                              const std::vector<MutationTask> &tasks,
                              double mutRate, double mutRange,
                              std::atomic<bool> *finished = nullptr);

  static unsigned optimalThreadCount();
  unsigned indexOfBest() const;

  // --- Data ---
  std::unique_ptr<Net *[]> nets;
  std::unique_ptr<int[]> scores;
  unsigned populationSize;
  unsigned evolutionCount = 0;
  double temperature;
  bool useThreads;
};

#endif // POPULATION_H
