#ifndef NEURON_H
#define NEURON_H

#include <random>
#include <vector>


#include <random>

// extern std::mt19937/*minstd_rand*/ generator;
// extern std::uniform_real_distribution<double> distribution;

// extern std::mt19937 gen;
// extern std::normal_distribution<double> randomGaussianDistribution;

#include <cstdint>

class FastRandom {
public:
  using result_type = std::uint64_t;

  FastRandom(result_type seed = 0) : state{seed, 0} {}

  result_type operator()() {
    result_type x = state[0];
    result_type const y = state[1];
    state[0] = y;
    x ^= x << 23;                             // a
    state[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
    return state[1] + y;
  }

  static constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }
  static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }

private:
  result_type state[2];
};

class Neuron;
typedef Neuron **Layer;

struct LayerType {

  enum Aggregationsfunktion { SUM = 0, AVG, MAX, MIN, INPUT_LAYER } aggrF;

  enum Aktivierungsfunktion {
    TANH = 0,
    RELU,
    SMAX,
    NONE,
    IDENTITY,
    SOFTPLUS,
    LEAKYRELU,
    SIGMOID
  } aktiF;

  unsigned neuronCount;

  LayerType() {}
  LayerType(unsigned neuronCount, Aggregationsfunktion aggrF,
            Aktivierungsfunktion aktiF)
      : aggrF(aggrF), aktiF(aktiF), neuronCount(neuronCount) {}

  std::string aggregationsfunktionToString();

  std::string aktivierungsfunktionToString();
};

class Neuron {
public:
  Neuron(unsigned anzahl_an_neuronen_des_naechsten_Layers, unsigned my_Index,
         const LayerType &layerT, const double &init_range = 1.0,
         bool enable_batch_learning = true);
  ~Neuron();

  // feed forward
  void aggregation(const Layer &prevLayer, const unsigned &neuron_count);

  void activation(const double &logsumexp = 0.0);
  // change weigts for genetic algorithm
  void mutate(const double &rate, const double &m_range);
  // get output

  double getOutputVal() const;

  // gradien calculation for feed forward back probagation
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer &nextLayer,
                           unsigned neuroncount_with_bias);
  void updateInputWeights(Layer &prevLayer,
                          const unsigned int &prevLayerNeuronCount,
                          const double &eta, const double &alpha,
                          const bool batchLearning);
  double sumDW(const Layer &nextLayer, unsigned neuroncount_with_bias) const;

  void calcBatchWeights(Layer &prevLayer,
                        const unsigned int &prevLayerNeuronCount,
                        const double &eta, const double &alpha);
  void applyBatch();

  // intern
  unsigned int getConections_count() const;
  double getSumme_Aggregationsfunktion() const;
  char getType() const;

  void setOutputVal(const double &newOutputVal);
  double *m_outputWeights;
  double *delta_Weights;
  double *batch_weight;
  bool batch_learning_enabled;

  LayerType::Aggregationsfunktion getAggrF() const;
  LayerType::Aktivierungsfunktion getAktiF() const;

private:
  double activationFunction(const double &x, const double &logsumexp = 0.0);
  double activationFunctionDerative(const double &x) const;

  double randomWeight();
  LayerType layerType;

  unsigned conections_count;
  double result_Aggregationsfunktion = 0;
  double m_outputVal = 0.0;
  unsigned my_Index;
  double m_gradient;
  double logsumexp;

  double softmax(const double &x) const;
  static double sigmoid(const double &x);
};

#endif // NEURON_H
