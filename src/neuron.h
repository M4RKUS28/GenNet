#ifndef NEURON_H
#define NEURON_H

#include "fastrandom.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LayerType — describes the configuration for one network layer
// ---------------------------------------------------------------------------

struct LayerType {
  enum AggregationFunction { SUM = 0, AVG, MAX, MIN, INPUT_LAYER } aggregation;

  enum ActivationFunction {
    TANH = 0,
    RELU,
    SMAX,
    NONE,
    IDENTITY,
    SOFTPLUS,
    LEAKYRELU,
    SIGMOID
  } activation;

  unsigned neuronCount = 0;

  LayerType() : aggregation(INPUT_LAYER), activation(NONE), neuronCount(0) {}
  LayerType(unsigned neuronCount, AggregationFunction aggr,
            ActivationFunction acti)
      : aggregation(aggr), activation(acti), neuronCount(neuronCount) {}

  std::string aggregationToString() const;
  std::string activationToString() const;
};

// ---------------------------------------------------------------------------
// Neuron
// ---------------------------------------------------------------------------

class Neuron;
typedef Neuron **Layer;

class Neuron {
public:
  Neuron(unsigned nextLayerSize, unsigned myIndex, const LayerType &layerT,
         double initRange = 1.0, bool enableBatchLearning = true);
  ~Neuron();

  // Non-copyable, non-movable (owned by Net via raw arrays)
  Neuron(const Neuron &) = delete;
  Neuron &operator=(const Neuron &) = delete;
  Neuron(Neuron &&) = delete;
  Neuron &operator=(Neuron &&) = delete;

  // --- Forward pass ---
  void aggregation(const Layer &prevLayer, unsigned neuronCount);
  void activation(double logsumexp = 0.0);

  // --- Mutation (genetic algorithms) ---
  void mutate(double rate, double range);

  // --- Backpropagation ---
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer,
                           unsigned neuronCountWithBias);
  void updateInputWeights(Layer &prevLayer, unsigned prevLayerNeuronCount,
                          double eta, double alpha, bool batchLearning);
  double sumDW(const Layer &nextLayer, unsigned neuronCountWithBias) const;

  void calcBatchWeights(Layer &prevLayer, unsigned prevLayerNeuronCount,
                        double eta, double alpha);
  void applyBatch();

  // --- Accessors ---
  double getOutputVal() const { return m_outputVal; }
  void setOutputVal(double val) { m_outputVal = val; }
  unsigned getConnectionCount() const { return connectionCount; }
  double getAggregationResult() const { return aggregationResult; }
  char getType() const;

  LayerType::AggregationFunction getAggregation() const {
    return layerType.aggregation;
  }
  LayerType::ActivationFunction getActivation() const {
    return layerType.activation;
  }

  // Weight access — needed by Net for save/load/copy
  double getWeight(unsigned idx) const { return weights[idx]; }
  void setWeight(unsigned idx, double val) { weights[idx] = val; }
  double getDeltaWeight(unsigned idx) const { return deltaWeights[idx]; }
  void setDeltaWeight(unsigned idx, double val) { deltaWeights[idx] = val; }
  void addToBatchWeight(unsigned idx, double val) { batchWeights[idx] += val; }

private:
  double activationFunction(double x, double logsumexp = 0.0);
  double activationFunctionDerivative(double x) const;
  double softmax(double x) const;
  static double sigmoid(double x);
  double randomWeight();

  // --- Weights (owned arrays) ---
  std::unique_ptr<double[]> weights;
  std::unique_ptr<double[]> deltaWeights;
  std::unique_ptr<double[]> batchWeights; // nullptr if batch learning disabled

  // --- State ---
  LayerType layerType;
  unsigned connectionCount;
  double aggregationResult = 0.0;
  double m_outputVal = 0.0;
  unsigned myIndex;
  double m_gradient = 0.0;
  double m_logsumexp = 0.0;
  bool batchLearningEnabled;

  // --- RNG (thread-local, defined in .cpp) ---
  // Uses global thread_local FastRandom + distributions
};

#endif // NEURON_H
