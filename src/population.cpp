#include "population.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#ifdef _WIN32
#include <windows.h>
static void sleepMs(unsigned ms) { Sleep(ms); }
#else
#include <unistd.h>
static void sleepMs(unsigned ms) { usleep(ms * 1000); }
#endif

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

Population::Population(const std::string &topology, unsigned size,
                       double initRange, double initTemperature,
                       bool useMutationThreads)
    : nets(std::make_unique<Net *[]>(size)),
      scores(std::make_unique<int[]>(size)), populationSize(size),
      temperature(initTemperature), useThreads(useMutationThreads) {
  if (size == 0) {
    std::cerr << "Population size must be > 0" << std::endl;
    return;
  }

  std::vector<LayerType> top = Net::getTopologyFromStr(topology);
  for (unsigned i = 0; i < populationSize; i++) {
    nets[i] = new Net(top, initRange);
    scores[i] = 0;
  }
}

Population::~Population() {
  for (unsigned i = 0; i < populationSize; i++) {
    delete nets[i];
  }
  // unique_ptr<Net*[]> and unique_ptr<int[]> handle array deallocation
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

Net *Population::netAt(unsigned index) { return nets[index]; }
const Net *Population::netAt(unsigned index) const { return nets[index]; }

unsigned Population::indexOfBest() const {
  int maxScore = scores[0];
  unsigned bestIdx = 0;
  for (unsigned i = 1; i < populationSize; ++i) {
    if (scores[i] > maxScore) {
      maxScore = scores[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

unsigned Population::optimalThreadCount() {
  unsigned hw = std::thread::hardware_concurrency();
  return (hw > 1) ? hw - 1 : 1;
}

// ---------------------------------------------------------------------------
// Evolution (copy best + mutate)
// ---------------------------------------------------------------------------

void Population::evolve(unsigned bestIdx, double mutationRate,
                        double mutationRange) {
  if (bestIdx >= populationSize || populationSize == 0) {
    std::cerr << "evolve(): invalid bestIdx or population size" << std::endl;
    return;
  }

  std::vector<std::pair<std::thread, std::atomic<bool> *>> mutThreads;
  std::vector<MutationTask> tasks;
  unsigned threadCount = optimalThreadCount();

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<unsigned> indexDist(0, populationSize - 1);

  evolutionCount++;

  double quarter = static_cast<double>(populationSize) / 4.0;
  unsigned taskSplitSize = std::max(1u, populationSize / threadCount);

  std::cout << "[0] --> < [" << static_cast<unsigned>(3.0 * quarter) << "]; ["
            << static_cast<unsigned>(3.0 * quarter) << "] --> < ["
            << static_cast<unsigned>(3.0 * quarter + quarter / 2.0)
            << "]; [*] --> < [" << populationSize << "]" << std::endl;

  // Group 1: 75% of population — copy from best, standard mutation
  for (unsigned i = 0; i < static_cast<unsigned>(3.0 * quarter); i++) {
    if (i == bestIdx)
      continue;
    tasks.push_back({static_cast<int>(bestIdx), static_cast<int>(i)});
    if (tasks.size() >= taskSplitSize) {
      dispatchMutation(mutThreads, tasks, threadCount, mutationRate,
                       mutationRange);
      tasks.clear();
    }
  }

  // Group 2: 12.5% — copy from best, low mutation
  for (unsigned i = static_cast<unsigned>(3.0 * quarter);
       i < static_cast<unsigned>(3.0 * quarter + quarter / 2.0); i++) {
    unsigned num = indexDist(rng);
    if (i == bestIdx || num == i)
      continue;
    tasks.push_back({static_cast<int>(bestIdx), static_cast<int>(i)});
    if (tasks.size() >= taskSplitSize) {
      dispatchMutation(mutThreads, tasks, threadCount, 0.01, 0.2);
      tasks.clear();
    }
  }

  // Group 3: 12.5% — copy from best, high mutation
  for (unsigned i = static_cast<unsigned>(3.0 * quarter + quarter / 2.0);
       i < populationSize; i++) {
    if (i == bestIdx)
      continue;
    tasks.push_back({static_cast<int>(bestIdx), static_cast<int>(i)});
    if (tasks.size() >= taskSplitSize) {
      dispatchMutation(mutThreads, tasks, threadCount, 0.03, 0.3);
      tasks.clear();
    }
  }

  // Dispatch remaining tasks
  if (!tasks.empty()) {
    dispatchMutation(mutThreads, tasks, threadCount, 0.01, 0.2);
    tasks.clear();
  }

  // Log mutation diversity
  double diffs[3] = {};
  if (populationSize >= 3) {
    diffs[0] = nets[1]->getDifferenceFrom(nets[bestIdx]);
    diffs[1] = nets[populationSize - 1]->getDifferenceFrom(nets[bestIdx]);
    diffs[2] = nets[populationSize / 2 + 1]->getDifferenceFrom(nets[bestIdx]);
  }

  // Join all threads
  for (auto &pair : mutThreads) {
    if (pair.first.joinable()) {
      pair.first.join();
      delete pair.second;
    }
  }

  if (populationSize >= 3) {
    std::cout << "Evo: " << evolutionCount << " Avg mutation: [1]=" << diffs[0]
              << ", [" << populationSize - 1 << "]=" << diffs[1] << ", ["
              << populationSize / 2 + 1 << "]=" << diffs[2]
              << " diff from best [" << bestIdx << "]" << std::endl;
  } else {
    std::cout << "Evo: " << evolutionCount << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Evolution with simulated annealing
// ---------------------------------------------------------------------------

void Population::evolveWithSimulatedAnnealing(double mutationRate,
                                              double mutationRange,
                                              double temperatureDecay) {
  if (populationSize == 0 || temperature == 0.0) {
    std::cerr << "evolveWithSimulatedAnnealing(): invalid state" << std::endl;
    return;
  }

  std::vector<std::pair<std::thread, std::atomic<bool> *>> mutThreads;
  std::vector<MutationTask> tasks;
  unsigned threadCount = optimalThreadCount();

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<unsigned> indexDist(0, populationSize - 1);
  std::uniform_real_distribution<double> probDist(0.0, 1.0);

  unsigned taskSplitSize = std::max(1u, populationSize / threadCount);

  evolutionCount++;
  temperature *= temperatureDecay;

  unsigned best = indexOfBest();
  int lastAcceptedWorse = -1;
  int worseAcceptedCount = 0;

  for (unsigned i = 0; i < populationSize; i++) {
    if (i != best) {
      if (scores[i] >= scores[best]) {
        // As good or better — keep and mutate in-place
        tasks.push_back({static_cast<int>(i), static_cast<int>(i)});
      } else if (scores[best] != 0) {
        // Simulated annealing acceptance probability
        double probability =
            std::exp(((static_cast<double>(scores[i]) / scores[best]) - 1.0) /
                     temperature);
        if (probDist(rng) < probability) {
          tasks.push_back({static_cast<int>(i), static_cast<int>(i)});
          lastAcceptedWorse = i;
          worseAcceptedCount++;
        } else {
          tasks.push_back({static_cast<int>(best), static_cast<int>(i)});
        }
      }
    }

    // Dispatch when batch is full or at end
    if (tasks.size() >= taskSplitSize || i == populationSize - 1) {
      if (!tasks.empty()) {
        dispatchMutation(mutThreads, tasks, threadCount,
                         mutationRate * temperature, mutationRange);
        tasks.clear();
      }
    }
  }

  // Join all threads
  for (auto &pair : mutThreads) {
    if (pair.first.joinable()) {
      pair.first.join();
      delete pair.second;
    }
  }

  // Logging
  if (lastAcceptedWorse >= 0 &&
      lastAcceptedWorse < static_cast<int>(populationSize)) {
    std::cout << " Avg mutation: [best] to [acceptedWorse] = "
              << nets[lastAcceptedWorse]->getDifferenceFrom(nets[best])
              << std::endl;
  }
  std::cout << " Temperature: " << temperature << " | 50% rejection threshold: "
            << temperature * (std::log(0.5) + 1.0)
            << " % of MaxScore=" << scores[best] << ", i.e. "
            << (temperature * (std::log(0.5) + 1.0)) * scores[best]
            << std::endl;
  std::cout << " Worse-accepted: "
            << static_cast<int>(100.0 * worseAcceptedCount / populationSize)
            << "% (" << worseAcceptedCount << ")" << std::endl;
  std::cout << " Effective mutation rate: " << temperature * mutationRate
            << std::endl;
  std::cout << "Finished evolution: " << evolutionCount << std::endl;
}

// ---------------------------------------------------------------------------
// Thread dispatch
// ---------------------------------------------------------------------------

void Population::dispatchMutation(
    std::vector<std::pair<std::thread, std::atomic<bool> *>> &threads,
    const std::vector<MutationTask> &tasks, unsigned maxThreads, double mutRate,
    double mutRange) {
  if (tasks.empty())
    return;

  if (useThreads) {
    // Wait until a thread slot is available
    while (threads.size() >= maxThreads) {
      for (auto it = threads.begin(); it != threads.end();) {
        if (*it->second) {
          if (it->first.joinable())
            it->first.join();
          delete it->second;
          it = threads.erase(it);
        } else {
          ++it;
        }
      }
      if (threads.size() >= maxThreads)
        sleepMs(50);
    }

    auto *finished = new std::atomic<bool>(false);
    threads.emplace_back(std::thread(executeMutation, nets.get(), tasks,
                                     mutRate, mutRange, finished),
                         finished);
  } else {
    executeMutation(nets.get(), tasks, mutRate, mutRange);
  }
}

void Population::executeMutation(Net **nets,
                                 const std::vector<MutationTask> &tasks,
                                 double mutRate, double mutRange,
                                 std::atomic<bool> *finished) {
  for (const auto &task : tasks) {
    if (!nets[task.targetIdx]->createCopyFrom(nets[task.sourceIdx]))
      std::cerr << "createCopyFrom failed: " << task.sourceIdx << " -> "
                << task.targetIdx << std::endl;
    else
      nets[task.targetIdx]->mutate(mutRate, mutRange);
  }
  if (finished)
    *finished = true;
}
