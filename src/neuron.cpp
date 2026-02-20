#include "neuron.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// ---------------------------------------------------------------------------
// Thread-local RNG instances (safe for multi-threaded Population mutation)
// ---------------------------------------------------------------------------

static thread_local FastRandom tl_uniformRng(std::random_device{}());
static thread_local std::uniform_real_distribution<double> tl_uniformDist(-1.0,
                                                                          1.0);
static thread_local FastRandom tl_gaussianRng(std::random_device{}());
static thread_local std::normal_distribution<double> tl_gaussianDist(0.0, 1.0);

// ---------------------------------------------------------------------------
// Neuron construction / destruction
// ---------------------------------------------------------------------------

Neuron::Neuron(unsigned nextLayerSize, unsigned myIndex,
               const LayerType &layerT, double initRange,
               bool enableBatchLearning)
    : weights(std::make_unique<double[]>(nextLayerSize)),
      deltaWeights(std::make_unique<double[]>(nextLayerSize)),
      batchWeights(enableBatchLearning
                       ? std::make_unique<double[]>(nextLayerSize)
                       : nullptr),
      layerType(layerT), connectionCount(nextLayerSize), myIndex(myIndex),
      batchLearningEnabled(enableBatchLearning) {
  for (unsigned c = 0; c < nextLayerSize; ++c) {
    weights[c] = randomWeight() * initRange;
    deltaWeights[c] = 0.0;
    if (batchWeights)
      batchWeights[c] = 0.0;
  }
}

Neuron::~Neuron() = default;

// ---------------------------------------------------------------------------
// Random weight generation
// ---------------------------------------------------------------------------

double Neuron::randomWeight() { return tl_uniformDist(tl_uniformRng); }

// ---------------------------------------------------------------------------
// Mutation (genetic algorithms)
// ---------------------------------------------------------------------------

void Neuron::mutate(double rate, double range) {
  for (unsigned i = 0; i < connectionCount; i++) {
    if (std::abs(tl_uniformDist(tl_uniformRng)) < rate) {
      weights[i] += tl_gaussianDist(tl_gaussianRng) * range;
      weights[i] = std::clamp(weights[i], -1.0, 1.0);
    }
  }
}

// ---------------------------------------------------------------------------
// Forward pass: aggregation + activation
// ---------------------------------------------------------------------------

void Neuron::aggregation(const Layer &prevLayer, unsigned neuronCount) {
  aggregationResult = 0.0;

  switch (getAggregation()) {
  case LayerType::SUM:
    for (unsigned n = 0; n < neuronCount; ++n)
      aggregationResult +=
          prevLayer[n]->getOutputVal() * prevLayer[n]->getWeight(myIndex);
    break;

  case LayerType::AVG:
    for (unsigned n = 0; n < neuronCount; ++n)
      aggregationResult +=
          prevLayer[n]->getOutputVal() * prevLayer[n]->getWeight(myIndex);
    aggregationResult /= static_cast<double>(neuronCount);
    break;

  case LayerType::MAX:
    aggregationResult = -std::numeric_limits<double>::infinity();
    for (unsigned n = 0; n < neuronCount; ++n)
      aggregationResult =
          std::max(aggregationResult, prevLayer[n]->getOutputVal() *
                                          prevLayer[n]->getWeight(myIndex));
    break;

  case LayerType::MIN:
    aggregationResult = std::numeric_limits<double>::infinity();
    for (unsigned n = 0; n < neuronCount; ++n)
      aggregationResult =
          std::min(aggregationResult, prevLayer[n]->getOutputVal() *
                                          prevLayer[n]->getWeight(myIndex));
    break;

  case LayerType::INPUT_LAYER:
    break;
  }
}

void Neuron::activation(double logsumexp) {
  m_outputVal = activationFunction(aggregationResult, logsumexp);
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

double Neuron::activationFunction(double x, double logsumexp) {
  switch (getActivation()) {
  case LayerType::TANH:
    return std::tanh(x);
  case LayerType::RELU:
    return std::max(0.0, x);
  case LayerType::SMAX:
    m_logsumexp = logsumexp;
    return softmax(x);
  case LayerType::NONE:
    return x;
  case LayerType::IDENTITY:
    return x;
  case LayerType::SOFTPLUS:
    return std::log(1.0 + std::exp(x));
  case LayerType::LEAKYRELU:
    return std::max(x, 0.1 * x);
  case LayerType::SIGMOID:
    return sigmoid(x);
  }
  return -1.0;
}

double Neuron::activationFunctionDerivative(double x) const {
  switch (getActivation()) {
  case LayerType::TANH:
    return 1.0 - x * x;
  case LayerType::RELU:
    return (x > 0) ? 1.0 : 0.0;
  case LayerType::SMAX: {
    double s = softmax(x);
    return s * (1.0 - s);
  }
  case LayerType::NONE:
    return 1.0; // d/dx(x) = 1  (was incorrectly 0.0)
  case LayerType::IDENTITY:
    return 1.0; // d/dx(x) = 1  (was incorrectly x)
  case LayerType::SOFTPLUS:
    return sigmoid(x);
  case LayerType::LEAKYRELU:
    return (x > 0) ? 1.0 : 0.1;
  case LayerType::SIGMOID: {
    double s = sigmoid(x);
    return s * (1.0 - s);
  }
  }
  return 1.0;
}

double Neuron::softmax(double x) const {
  double val = std::exp(x - m_logsumexp);
  if (std::isinf(val) || std::isnan(val)) {
    std::cerr << "Softmax: invalid value (inf/nan) for x=" << x
              << " logsumexp=" << m_logsumexp << std::endl;
    return 0.0;
  }
  return val;
}

double Neuron::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---------------------------------------------------------------------------
// Backpropagation
// ---------------------------------------------------------------------------

void Neuron::calcOutputGradients(double targetVal) {
  double delta = targetVal - m_outputVal;
  m_gradient = delta * activationFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer,
                                 unsigned neuronCountWithBias) {
  double dow = sumDW(nextLayer, neuronCountWithBias);
  m_gradient = dow * activationFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer, unsigned prevLayerNeuronCount,
                                double eta, double alpha, bool batchLearning) {
  for (unsigned n = 0; n < prevLayerNeuronCount; ++n) {
    Neuron *neuron = prevLayer[n];

    double oldDelta = neuron->getDeltaWeight(myIndex);
    double newDelta =
        eta * neuron->getOutputVal() * m_gradient + alpha * oldDelta;

    neuron->setDeltaWeight(myIndex, newDelta);
    if (batchLearning) {
      if (!batchLearningEnabled) {
        std::cerr << "Error: batch learning not enabled!" << std::endl;
        return;
      }
      neuron->addToBatchWeight(myIndex, newDelta);
    } else {
      neuron->setWeight(myIndex, neuron->getWeight(myIndex) + newDelta);
    }
  }
}

double Neuron::sumDW(const Layer &nextLayer,
                     unsigned neuronCountWithBias) const {
  double sum = 0.0;
  for (unsigned n = 0; n < neuronCountWithBias; ++n) {
    sum += weights[n] * nextLayer[n]->m_gradient;
  }
  return sum;
}

void Neuron::applyBatch() {
  if (!batchLearningEnabled || !batchWeights) {
    std::cerr << "Error: batch learning not enabled!" << std::endl;
    return;
  }
  for (unsigned i = 0; i < connectionCount; ++i) {
    weights[i] += batchWeights[i];
    batchWeights[i] = 0.0;
  }
}

char Neuron::getType() const {
  // Placeholder — return 'N' for normal neuron
  return 'N';
}

// ---------------------------------------------------------------------------
// LayerType string conversion
// ---------------------------------------------------------------------------

std::string LayerType::aggregationToString() const {
  switch (aggregation) {
  case SUM:
    return "SUM";
  case AVG:
    return "AVG";
  case MAX:
    return "MAX";
  case MIN:
    return "MIN";
  case INPUT_LAYER:
    return "INPUT_LAYER";
  }
  return "Unknown";
}

std::string LayerType::activationToString() const {
  switch (activation) {
  case TANH:
    return "TANH";
  case RELU:
    return "RELU";
  case SMAX:
    return "SMAX";
  case NONE:
    return "NONE";
  case IDENTITY:
    return "IDENTITY";
  case SOFTPLUS:
    return "SOFTPLUS";
  case LEAKYRELU:
    return "LEAKYRELU";
  case SIGMOID:
    return "SIGMOID";
  }
  return "Unknown";
}
