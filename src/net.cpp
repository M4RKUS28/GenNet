#include "net.h"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Net::Net(const std::vector<LayerType> &topology, double initRange,
         bool enableBatchLearning)
    : topology(topology) {
  init(initRange, enableBatchLearning);
}

Net::Net(const std::string &s_topology, double initRange,
         bool enableBatchLearning)
    : topology(getTopologyFromStr(s_topology)) {
  init(initRange, enableBatchLearning);
}

Net::Net(const Net *other, double initRange, bool enableBatchLearning)
    : Net(other->getTopology()) {
  init(initRange, enableBatchLearning);
  createCopyFrom(other);
}

Net::Net(const std::string filename, bool &ok, bool enableBatchLearning) {
  ok = loadFrom(filename, enableBatchLearning);
}

void Net::init(double initRange, bool enableBatchLearning) {
  if (topology.size() < 2) {
    std::cerr << "Invalid net: topology size is too small: " << topology.size()
              << std::endl;
    return;
  }

  // Clean up any previously allocated layers
  cleanup();

  m_layers = new Layer[topology.size()];

  for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
    unsigned neuronCount = topology[layerNum].neuronCount + 1; // +1 for bias

    m_layers[layerNum] = new Neuron *[neuronCount];
    unsigned nextLayerSize = (layerNum == topology.size() - 1)
                                 ? 0
                                 : topology[layerNum + 1].neuronCount;

    for (unsigned neuronNum = 0; neuronNum < neuronCount; ++neuronNum) {
      m_layers[layerNum][neuronNum] =
          new Neuron(nextLayerSize, neuronNum, topology[layerNum], initRange,
                     enableBatchLearning);
    }

    // Set bias neuron output to 1.0
    m_layers[layerNum][topology[layerNum].neuronCount]->setOutputVal(1.0);
  }

  m_recentAverageSmoothingFactor = 0.0009;
  m_recentAverageError = 0.0;
}

Net::~Net() { cleanup(); }

void Net::cleanup() {
  if (!m_layers)
    return;

  for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
    unsigned neuronCount = topology[layerNum].neuronCount + 1; // +1 for bias
    for (unsigned neuronNum = 0; neuronNum < neuronCount; ++neuronNum) {
      delete m_layers[layerNum][neuronNum];
    }
    delete[] m_layers[layerNum];
  }
  delete[] m_layers;
  m_layers = nullptr;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

bool Net::saveTo(const std::string &filename) {
  std::ofstream outputFile(filename);
  if (!outputFile.is_open()) {
    std::cerr << "Unable to open file " << filename << std::endl;
    return false;
  }

  // Line 1: topology string
  outputFile << getTopologyStr() << "\n------\n";

  // Weights per layer (except output layer which has no outgoing connections)
  for (unsigned layer = 0; layer < layerCount() - 1; ++layer) {
    for (unsigned neuron = 0; neuron < neuronCountAt(layer) + 1; ++neuron) {
      unsigned conns = m_layers[layer][neuron]->getConnectionCount();
      for (unsigned w = 0; w < conns; ++w) {
        outputFile << m_layers[layer][neuron]->getWeight(w);
        if (w != conns - 1)
          outputFile << ",";
      }
      outputFile << "\n";
    }
    outputFile << "\n";
  }

  outputFile.close();
  std::cout << "Neural network saved to " << filename << std::endl;
  return true;
}

bool Net::loadFrom(const std::string &filename, bool enableBatchLearning) {
  std::ifstream inputFile(filename);
  if (!inputFile.is_open()) {
    std::cerr << "Unable to open file " << filename << std::endl;
    return false;
  }

  std::string topLines[2];
  if (!std::getline(inputFile, topLines[0]) ||
      !std::getline(inputFile, topLines[1])) {
    std::cerr << "Error: reading topology line" << std::endl;
    return false;
  }
  if (topLines[0].empty()) {
    std::cerr << "No topology found! Invalid file!" << std::endl;
    return false;
  }

  // Initialize network from file topology if not yet initialized
  if (!topology.empty()) {
    if (getTopologyStr() != topLines[0]) {
      std::cerr << "Topology mismatch! Existing: (" << getTopologyStr()
                << ") File: (" << topLines[0] << ")" << std::endl;
      return false;
    }
    // Topology matches — just load weights into existing network
  } else {
    // Not yet initialized — parse topology and create network
    this->topology = getTopologyFromStr(topLines[0]);
    if (topology.empty()) {
      std::cerr << "Invalid topology in file!" << std::endl;
      return false;
    }
    init(0.0, enableBatchLearning);
  }

  // Load weights
  for (unsigned layer = 0; layer < layerCount() - 1; ++layer) {
    for (unsigned neuron = 0; neuron < neuronCountAt(layer) + 1; ++neuron) {
      std::string line;
      if (!std::getline(inputFile, line)) {
        std::cerr << "Error: reading weight line" << std::endl;
        return false;
      }

      std::vector<double> weights;
      std::istringstream iss(line);
      double value;
      while (iss >> value) {
        weights.push_back(value);
        if (iss.peek() == ',')
          iss.ignore();
      }

      if (weights.size() != topology[layer + 1].neuronCount) {
        std::cerr << "Weight count mismatch: " << weights.size() << " vs "
                  << topology[layer + 1].neuronCount << std::endl;
        return false;
      }

      for (unsigned w = 0; w < weights.size(); ++w) {
        m_layers[layer][neuron]->setWeight(w, weights[w]);
      }
    }

    // Skip blank line between layers
    std::string blank;
    if (!std::getline(inputFile, blank)) {
      std::cerr << "Error: reading blank separator line" << std::endl;
      return false;
    }
  }

  inputFile.close();
  std::cout << "Neural network loaded from " << filename << std::endl;
  return true;
}

// ---------------------------------------------------------------------------
// Topology parsing
// ---------------------------------------------------------------------------

std::vector<LayerType> Net::getTopologyFromStr(const std::string &top) {
  std::istringstream stream(top);
  std::string substring;
  std::vector<LayerType> result;

  while (std::getline(stream, substring, ',')) {
    int count = 0;
    size_t start = std::string::npos;
    LayerType lt;
    substring += '_'; // sentinel for parsing

    for (size_t index = substring.find('_'); index != std::string::npos;
         index = substring.find('_', start + 1)) {
      std::string part = substring.substr(
          (start == std::string::npos) ? 0 : start + 1,
          index - ((start == std::string::npos) ? 0 : start + 1));
      if (part.empty()) {
        std::cerr << "Invalid argument: " << substring << std::endl;
        break;
      }

      if (count == 0) {
        // Neuron count
        try {
          lt.neuronCount = static_cast<unsigned>(std::abs(std::stoi(part)));
        } catch (const std::invalid_argument &e) {
          std::cerr << "Invalid argument: " << e.what() << std::endl;
          break;
        } catch (const std::out_of_range &e) {
          std::cerr << "Out of range: " << e.what() << std::endl;
          break;
        }

      } else if (count == 1) {
        // Aggregation function
        if (part == "SUM")
          lt.aggregation = LayerType::SUM;
        else if (part == "MAX")
          lt.aggregation = LayerType::MAX;
        else if (part == "MIN")
          lt.aggregation = LayerType::MIN;
        else if (part == "AVG")
          lt.aggregation = LayerType::AVG;
        else if (part == "INPUT") {
          lt.aggregation = LayerType::INPUT_LAYER;
          count = 3; // skip activation for input layer
          break;
        } else {
          std::cerr << "Invalid aggregation function: " << part << std::endl;
          break;
        }

      } else if (count == 2) {
        // Activation function
        if (part == "TANH")
          lt.activation = LayerType::TANH;
        else if (part == "RELU")
          lt.activation = LayerType::RELU;
        else if (part == "SMAX")
          lt.activation = LayerType::SMAX;
        else if (part == "LAYER") // INPUT_LAYER
          lt.activation = LayerType::NONE;
        else if (part == "IDENTITY")
          lt.activation = LayerType::IDENTITY;
        else if (part == "SOFTPLUS")
          lt.activation = LayerType::SOFTPLUS;
        else if (part == "LEAKYRELU")
          lt.activation = LayerType::LEAKYRELU;
        else if (part == "SIGMOID")
          lt.activation = LayerType::SIGMOID;
        else {
          std::cerr << "Invalid activation function: " << part << std::endl;
          break;
        }

      } else {
        std::cerr << "Invalid input — more than 2 underscores: " << substring
                  << std::endl;
        break;
      }

      count++;
      start = index;
    }

    if (count != 3) {
      std::cerr << "Invalid layer specification: " << substring << std::endl;
      continue;
    }
    result.push_back(lt);
  }

  return result;
}

std::string Net::getTopologyStr() const {
  std::string result;
  for (unsigned i = 0; i < topology.size(); ++i) {
    result += std::to_string(topology[i].neuronCount) + "_" +
              topology[i].aggregationToString() + "_" +
              topology[i].activationToString();
    if (i != topology.size() - 1)
      result += ",";
  }
  return result;
}

const std::vector<LayerType> &Net::getTopology() const { return topology; }

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

void Net::feedForward(const double *input) {
  // Set input layer values
  for (unsigned i = 0; i < neuronCountAt(0); i++) {
    m_layers[0][i]->setOutputVal(input[i]);
  }

  // Propagate through hidden and output layers
  for (unsigned layerNum = 1; layerNum < layerCount(); ++layerNum) {
    Layer &prevLayer = m_layers[layerNum - 1];
    Layer &thisLayer = m_layers[layerNum];

    bool isSoftmax = thisLayer[0]->getActivation() == LayerType::SMAX;
    double maxActivation = -std::numeric_limits<double>::infinity();
    double logSumExp = 0.0;

    // Aggregation
    for (unsigned n = 0; n < neuronCountAt(layerNum); ++n) {
      thisLayer[n]->aggregation(prevLayer, neuronCountAt(layerNum - 1) + 1);
    }

    // Numerically stable softmax preparation
    if (isSoftmax) {
      for (unsigned n = 0; n < neuronCountAt(layerNum); ++n)
        maxActivation =
            std::max(maxActivation, thisLayer[n]->getAggregationResult());
      for (unsigned n = 0; n < neuronCountAt(layerNum); ++n)
        logSumExp +=
            std::exp(thisLayer[n]->getAggregationResult() - maxActivation);
      logSumExp = maxActivation + std::log(logSumExp);
    }

    // Activation
    for (unsigned n = 0; n < neuronCountAt(layerNum); ++n) {
      thisLayer[n]->activation(logSumExp);
    }
  }
}

void Net::getResults(double *output) const {
  unsigned outputLayer = layerCount() - 1;
  for (unsigned n = 0; n < neuronCountAt(outputLayer); ++n) {
    output[n] = m_layers[outputLayer][n]->getOutputVal();
  }
}

// ---------------------------------------------------------------------------
// Backpropagation
// ---------------------------------------------------------------------------

void Net::backProp(double *targetVals, double eta, double alpha,
                   bool batchLearning) {
  Layer &outputLayer = m_layers[layerCount() - 1];
  unsigned outputNeuronCount = neuronCountAt(layerCount() - 1);

  // Calculate overall net error (RMS)
  m_error = 0.0;
  for (unsigned n = 0; n < outputNeuronCount; ++n) {
    double delta = targetVals[n] - outputLayer[n]->getOutputVal();
    m_error += delta * delta;
  }
  m_error /= static_cast<double>(outputNeuronCount);
  m_error = std::sqrt(m_error);
  m_recentAverageError =
      (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) /
      (m_recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients
  for (unsigned n = 0; n < outputNeuronCount; ++n) {
    outputLayer[n]->calcOutputGradients(targetVals[n]);
  }

  // Calculate hidden layer gradients (back to front, skip input layer)
  for (unsigned long layerNum = layerCount() - 2; layerNum > 0; --layerNum) {
    Layer &hiddenLayer = m_layers[layerNum];
    Layer &nextLayer = m_layers[layerNum + 1];
    for (unsigned n = 0; n < neuronCountAt(layerNum) + 1; ++n) {
      hiddenLayer[n]->calcHiddenGradients(nextLayer,
                                          neuronCountAt(layerNum + 1) + 1);
    }
  }

  // Update weights (back to front)
  for (long layerNum = layerCount() - 1; layerNum > 0; --layerNum) {
    Layer &layer = m_layers[layerNum];
    Layer &prevLayer = m_layers[layerNum - 1];
    for (unsigned n = 0; n < neuronCountAt(layerNum) + 1; ++n) {
      layer[n]->updateInputWeights(prevLayer, neuronCountAt(layerNum - 1) + 1,
                                   eta, alpha, batchLearning);
    }
  }
}

void Net::applyBatch() {
  // Apply accumulated batch weights (skip last layer)
  for (long layerNum = static_cast<long>(layerCount()) - 2; layerNum >= 0;
       --layerNum) {
    Layer &layer = m_layers[layerNum];
    for (unsigned n = 0; n < neuronCountAt(layerNum) + 1; ++n) {
      layer[n]->applyBatch();
    }
  }
}

// ---------------------------------------------------------------------------
// Genetic algorithms
// ---------------------------------------------------------------------------

void Net::mutate(double mutationRate, double mutationRange) {
  for (unsigned layerNum = 0; layerNum < topology.size() - 1; ++layerNum) {
    for (unsigned neuronNum = 0; neuronNum < topology[layerNum].neuronCount + 1;
         ++neuronNum) {
      m_layers[layerNum][neuronNum]->mutate(mutationRate, mutationRange);
    }
  }
}

bool Net::createCopyFrom(const Net *origin) {
  if (topology.size() != origin->topology.size()) {
    std::cerr << "Topology size mismatch: " << topology.size()
              << " != " << origin->topology.size() << std::endl;
    return false;
  }

  for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
    if (topology[layerNum].neuronCount !=
        origin->topology[layerNum].neuronCount) {
      std::cerr << "Neuron count mismatch at layer " << layerNum << ": "
                << topology[layerNum].neuronCount
                << " != " << origin->topology[layerNum].neuronCount
                << std::endl;
      return false;
    }

    for (unsigned neuronNum = 0; neuronNum < topology[layerNum].neuronCount + 1;
         ++neuronNum) {
      unsigned conns = m_layers[layerNum][neuronNum]->getConnectionCount();
      if (conns !=
          origin->m_layers[layerNum][neuronNum]->getConnectionCount()) {
        std::cerr << "Connection count mismatch" << std::endl;
        return false;
      }
      for (unsigned c = 0; c < conns; c++) {
        m_layers[layerNum][neuronNum]->setWeight(
            c, origin->m_layers[layerNum][neuronNum]->getWeight(c));
      }
    }
  }
  return true;
}

double Net::getDifferenceFrom(const Net *other) {
  if (other->getTopologyStr() != getTopologyStr()) {
    std::cerr << "Cannot compute difference: topology mismatch!" << std::endl;
    return -1.0;
  }

  double diff = 0.0;
  size_t weightCount = 0;

  for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
    for (unsigned neuronNum = 0; neuronNum < topology[layerNum].neuronCount + 1;
         ++neuronNum) {
      unsigned conns = m_layers[layerNum][neuronNum]->getConnectionCount();
      for (unsigned c = 0; c < conns; c++) {
        diff += std::abs(m_layers[layerNum][neuronNum]->getWeight(c) -
                         other->m_layers[layerNum][neuronNum]->getWeight(c));
        weightCount++;
      }
    }
  }

  return (weightCount > 0) ? diff / static_cast<double>(weightCount) : 0.0;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

double Net::getConnectionWeight(unsigned layer, unsigned neuronFrom,
                                unsigned neuronTo) const {
  if (layer >= layerCount() || neuronFrom >= neuronCountAt(layer) + 1 ||
      neuronTo >= m_layers[layer][neuronFrom]->getConnectionCount()) {
    std::cerr << "getConnectionWeight() invalid: " << layer << " " << neuronFrom
              << " " << neuronTo << std::endl;
    return -1.0;
  }
  return m_layers[layer][neuronFrom]->getWeight(neuronTo);
}

double Net::getNeuronValue(unsigned layer, unsigned neuron) const {
  if (layer >= layerCount() || neuron >= neuronCountAt(layer) + 1) {
    std::cerr << "getNeuronValue() invalid: " << layer << " " << neuron
              << std::endl;
    return -1.0;
  }
  return m_layers[layer][neuron]->getOutputVal();
}

unsigned Net::layerCount() const { return topology.size(); }

unsigned Net::neuronCountAt(unsigned layer) const {
  return topology[layer].neuronCount;
}

double Net::recentAverageError() const { return m_recentAverageError; }
