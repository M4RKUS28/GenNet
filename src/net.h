#ifndef NET_H
#define NET_H

#include "neuron.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

typedef Neuron **Layer;

class Net {
public:
  Net(const std::string &topology, double initRange = 0.01,
      bool enableBatchLearning = true);
  Net(const std::vector<LayerType> &topology, double initRange = 0.01,
      bool enableBatchLearning = true);
  Net(const Net *other, double initRange = 0.01,
      bool enableBatchLearning = true);
  Net(const std::string filename, bool &ok, bool enableBatchLearning = true);
  ~Net();

  // Non-copyable (use createCopyFrom for weight copying)
  Net(const Net &) = delete;
  Net &operator=(const Net &) = delete;

  // --- Serialization ---
  bool saveTo(const std::string &path);
  bool loadFrom(const std::string &path, bool enableBatchLearning = true);

  // --- Topology ---
  const std::vector<LayerType> &getTopology() const;
  static std::vector<LayerType> getTopologyFromStr(const std::string &top);
  std::string getTopologyStr() const;

  // --- Forward / Backward pass ---
  void feedForward(const double *input);
  void getResults(double *output) const;
  void backProp(double *targetVals, double eta, double alpha,
                bool batchLearning = false);
  void applyBatch();

  // --- Genetic algorithms ---
  void mutate(double mutationRate, double mutationRange = 1.0);
  bool createCopyFrom(const Net *origin);
  double getDifferenceFrom(const Net *other);

  // --- Accessors ---
  double getConnectionWeight(unsigned layer, unsigned neuronFrom,
                             unsigned neuronTo) const;
  double getNeuronValue(unsigned layer, unsigned neuron) const;
  double recentAverageError() const;

private:
  unsigned layerCount() const;
  unsigned neuronCountAt(unsigned layer) const;
  void init(double initRange = 1.0, bool enableBatchLearning = true);
  void cleanup();

  Layer *m_layers = nullptr;
  std::vector<LayerType> topology;

  double m_error = 0.0;
  double m_recentAverageSmoothingFactor = 0.0009;
  double m_recentAverageError = 0.0;
};

#endif // NET_H
