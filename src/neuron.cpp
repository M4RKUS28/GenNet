#include "neuron.h"
#include <cmath>
#include <iostream>
#include <ostream>

Neuron::Neuron(unsigned int anzahl_an_neuronen_des_naechsten_Layers,
               unsigned int my_Index, const LayerType &layerT,
               const double &init_range, bool enable_batch_learning)
    : batch_learning_enabled(enable_batch_learning), layerType(layerT),
      conections_count(anzahl_an_neuronen_des_naechsten_Layers),
      my_Index(my_Index), m_gradient(0.0)

{
  m_outputWeights = new double[anzahl_an_neuronen_des_naechsten_Layers];
  delta_Weights = new double[anzahl_an_neuronen_des_naechsten_Layers];
  if (batch_learning_enabled)
    batch_weight = new double[anzahl_an_neuronen_des_naechsten_Layers];

  for (unsigned c = 0; c < anzahl_an_neuronen_des_naechsten_Layers; ++c) {
    m_outputWeights[c] = randomWeight() * init_range;
    delta_Weights[c] = 0.0;
    if (batch_learning_enabled)
      batch_weight[c] = 0.0;
  }
}

Neuron::~Neuron() { delete m_outputWeights; }

FastRandom /*std::mt19937*/ /*minstd_rand*/ generator2(std::random_device{}());
std::uniform_real_distribution<double> distribution(-1.0, 1.0);

double Neuron::randomWeight() { return distribution(generator2); }

FastRandom /*std::mt19937*/ gen2(std::random_device{}());
std::normal_distribution<double> randomGaussianDistribution(0.0, 1.0);

void Neuron::mutate(const double &rate, const double &m_range) {
  for (unsigned i = 0; i < conections_count; i++)
    if (std::abs(distribution(generator2)) < rate) {
      m_outputWeights[i] += randomGaussianDistribution(gen2) * m_range;
      if (m_outputWeights[i] < -1)
        m_outputWeights[i] = -1;
      else if (m_outputWeights[i] > 1)
        m_outputWeights[i] = 1;
    }
}

void Neuron::activation(const double &logsumexp) {
  m_outputVal =
      activationFunction(this->result_Aggregationsfunktion, logsumexp);
}

void Neuron::aggregation(const Layer &prevLayer,
                         const unsigned int &neuron_count) {
  this->result_Aggregationsfunktion = 0.0;

  switch (this->getAggrF()) {
  case LayerType::Aggregationsfunktion::SUM:
    for (unsigned n = 0; n < neuron_count; ++n)
      this->result_Aggregationsfunktion +=
          prevLayer[n]->getOutputVal() *
          prevLayer[n]->m_outputWeights[my_Index];
    break;
  case LayerType::Aggregationsfunktion::AVG:
    for (unsigned n = 0; n < neuron_count; ++n)
      this->result_Aggregationsfunktion +=
          prevLayer[n]->getOutputVal() *
          prevLayer[n]->m_outputWeights[my_Index];
    this->result_Aggregationsfunktion /= (double)neuron_count;
    break;
  case LayerType::Aggregationsfunktion::MAX:
    for (unsigned n = 0; n < neuron_count; ++n)
      this->result_Aggregationsfunktion =
          std::max(this->result_Aggregationsfunktion,
                   prevLayer[n]->getOutputVal() *
                       prevLayer[n]->m_outputWeights[my_Index]);
    break;
  case LayerType::Aggregationsfunktion::MIN:
    for (unsigned n = 0; n < neuron_count; ++n)
      this->result_Aggregationsfunktion =
          std::min(this->result_Aggregationsfunktion,
                   prevLayer[n]->getOutputVal() *
                       prevLayer[n]->m_outputWeights[my_Index]);
    break;
  case LayerType::Aggregationsfunktion::INPUT_LAYER:
    break;
  }
}
LayerType::Aktivierungsfunktion Neuron::getAktiF() const {
  return layerType.aktiF;
}

LayerType::Aggregationsfunktion Neuron::getAggrF() const {
  return layerType.aggrF;
}

double Neuron::getOutputVal() const { return m_outputVal; }

void Neuron::setOutputVal(const double &newOutputVal) {
  m_outputVal = newOutputVal;
}

unsigned int Neuron::getConections_count() const { return conections_count; }

double Neuron::getSumme_Aggregationsfunktion() const {
  return result_Aggregationsfunktion;
}
#include "math.h"
double Neuron::activationFunction(const double &x, const double &logsumexp) {
  // if(((x < 0.03 && exp_sum < 0.03) && this->getAktiF() == 2) || x < 0.003 )
  //     std::cout << "x: " << x << " exp_sum: " << exp_sum << " neuron: " <<
  //     this->my_Index << " " << this->getAktiF() << std::endl;

  switch (this->getAktiF()) {
  case LayerType::Aktivierungsfunktion::TANH:
    return std::tanh(x);
  case LayerType::Aktivierungsfunktion::RELU:
    return std::max(0.0, x);
  case LayerType::Aktivierungsfunktion::SMAX:
    this->logsumexp = logsumexp;
    return softmax(x);
    /* The second case, −S(zi)×S(zj)−S(zi​)×S(zj​), is relevant in a
     * specific scenario where you're interested in how the output of one neuron
     * (neuron ii) affects the input of a different neuron (neuron jj). This
     * scenario arises in more advanced architectures like recurrent neural
     * networks (RNNs) or in cases where there are connections between neurons
     * that are not in adjacent layers. Here's a practical example: Consider a
     * recurrent neural network (RNN) with connections that loop back on
     * themselves. In this case, the output of a neuron at time step tt (let's
     * say neuron ii at time tt) can affect the input of the same neuron at the
     * next time step t+1t+1 (input zjzj​ of neuron ii at time t+1t+1).In this
     * scenario, if you're interested in computing the derivative of the output
     * of neuron ii at time tt with respect to the input of neuron ii at time
     * t+1t+1 (i.e., S(zi)∂zj∂zj​S(zi​)​), you would use the second
     * case:
     * S(zi)∂zj=−S(zi)×S(zj)∂zj​S(zi​)​=−S(zi​)×S(zj​) This situation
     * is specific to certain types of architectures, particularly those with
     * recurrent connections or non-standard connectivity patterns. In most
     * standard feedforward neural networks, you'll primarily use the first case
     * (S(zi)×(1−S(zi))S(zi​)×(1−S(zi​))) during backpropagation.*/
  case LayerType::Aktivierungsfunktion::NONE:
    return x;
  case LayerType::IDENTITY:
    return x;
  case LayerType::SOFTPLUS:
    return log(1.0 + exp(x));
  case LayerType::LEAKYRELU:
    return std::max(x, 0.1 * x);
  case LayerType::SIGMOID:
    return sigmoid(x);
  }
  return -1.0;
}

double Neuron::softmax(const double &x) const {
  double val = std::exp(x - logsumexp);
  if (val == std::numeric_limits<double>::infinity() ||
      val == std::numeric_limits<double>::signaling_NaN()) {
    perror("INVALID X VAL!");
    exit(12);
  }
  return val;
}

double Neuron::sigmoid(const double &x) { return 1.0 / (1.0 + exp(-x)); }

double Neuron::activationFunctionDerative(const double &x) const {
  switch (this->getAktiF()) {
  case LayerType::Aktivierungsfunktion::TANH:
    return 1.0 - x * x;
  case LayerType::Aktivierungsfunktion::RELU:
    return (x > 0) ? 1.0 : 0.0;
  case LayerType::Aktivierungsfunktion::SMAX: {
    double sigma_i = softmax(x); // /*softmax*/(std::exp(x) / denominator);
    return sigma_i * (1.0 - sigma_i) /** !ERRROR!   * denominator*/;
  }
  case LayerType::NONE:
    return 0.0;
  case LayerType::IDENTITY:
    return x;
  case LayerType::SOFTPLUS:
    return sigmoid(x);
  case LayerType::LEAKYRELU:
    return (x > 0) ? 1.0 : 0.1;
    break;
  case LayerType::SIGMOID: {
    double sig = sigmoid(x);
    return sig * (1 - sig);
  }
  }
  return x;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer,
                                 unsigned int neuroncount_with_bias) {
  double dow = sumDW(nextLayer, neuroncount_with_bias);
  m_gradient = dow * activationFunctionDerative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
  double delta = targetVal - m_outputVal;
  m_gradient = delta * activationFunctionDerative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer,
                                const unsigned &prevLayerNeuronCount,
                                const double &eta, const double &alpha,
                                const bool batchLearning) {

  for (unsigned n = 0; n < prevLayerNeuronCount; ++n) {
    Neuron *neuron = prevLayer[n];

    double oldDeltaWeight = neuron->delta_Weights[my_Index];
    double newDeltaWeight =
        eta * neuron->getOutputVal() * m_gradient +
        alpha * oldDeltaWeight; //<== HIER mit TZeit : 56:00 mint und
                                //https://www.youtube.com/watch?v=KkwX7FkLfug

    // Apply gradient clipping to newDeltaWeight

    // double clippingMaxVal = 1000.0;
    // if (newDeltaWeight > clippingMaxVal) {
    //     perror("WARNING: CLIPPING TO 1");
    //     newDeltaWeight = clippingMaxVal;
    // } else if (newDeltaWeight < -clippingMaxVal) {
    //     newDeltaWeight = -clippingMaxVal;
    //     perror("WARNING: CLIPPING TO -1");
    // }

    neuron->delta_Weights[(my_Index)] = newDeltaWeight;
    if (batchLearning) {
      if (!batch_learning_enabled) {
        perror("Batch learning not enabled!");
        return;
      } else {
        neuron->batch_weight[(my_Index)] += newDeltaWeight;
      }
    } else {
      neuron->m_outputWeights[(my_Index)] += newDeltaWeight;
    }
  }
}

double Neuron::sumDW(const Layer &nextLayer,
                     unsigned neuroncount_with_bias) const {
  double sum = 0.0;
  for (unsigned n = 0; n < neuroncount_with_bias; ++n) {
    sum += m_outputWeights[n] * nextLayer[n]->m_gradient;
  }
  return sum;
}

void Neuron::applyBatch() {
  if (!batch_learning_enabled) {
    perror("Batch learning not enabled!");
    return;
  }
  for (unsigned i = 0; i < this->getConections_count(); ++i) {
    this->m_outputWeights[i] += this->batch_weight[(i)];
    this->batch_weight[i] = 0.0;
  }
}

std::string LayerType::aggregationsfunktionToString() {
  if (aggrF == SUM)
    return "SUM";
  else if (aggrF == AVG)
    return "AVG";
  else if (aggrF == MAX)
    return "MAX";
  else if (aggrF == MIN)
    return "MIN";
  else if (aggrF == INPUT_LAYER)
    return "INPUT_LAYER";
  else
    return "Unknown";
}

std::string LayerType::aktivierungsfunktionToString() {
  if (aktiF == TANH)
    return "TANH";
  else if (aktiF == RELU)
    return "RELU";
  else if (aktiF == SMAX)
    return "SMAX";
  else if (aktiF == NONE)
    return "NONE";

  else if (aktiF == IDENTITY)
    return "IDENTITY";
  else if (aktiF == SOFTPLUS)
    return "SOFTPLUS";
  else if (aktiF == LEAKYRELU)
    return "LEAKYRELU";
  else if (aktiF == SIGMOID)
    return "SIGMOID";

  else
    return "Unknown";
}
