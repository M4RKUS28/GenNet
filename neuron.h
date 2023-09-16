#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>

#include <random>

extern std::mt19937/*minstd_rand*/ generator;
extern std::uniform_real_distribution<double> distribution;

extern std::mt19937 gen;
extern std::normal_distribution<double> randomGaussianDistribution;


class Neuron;
typedef  Neuron** Layer;


struct LayerType {

    enum Aggregationsfunktion  {
        SUM = 0,
        AVG,
        MAX,
        MIN,
    } aggrF;

    enum Aktivierungsfunktion  {
        TANH = 0,
        RELU,
        SMAX
    } aktiF;

    unsigned neuronCount;

    LayerType()  {}
    LayerType(unsigned neuronCount,  Aggregationsfunktion aggrF,   Aktivierungsfunktion aktiF)
        : aggrF(aggrF), aktiF(aktiF), neuronCount(neuronCount) { }

    std::string aggregationsfunktionToString() {
        if (aggrF == SUM) return "SUM";
        else if (aggrF == AVG) return "AVG";
        else if (aggrF == MAX) return "MAX";
        else if (aggrF == MIN) return "MIN";
        else return "Unknown";
    }

    std::string aktivierungsfunktionToString() {
        if (aktiF == TANH) return "TANH";
        else if (aktiF == RELU) return "RELU";
        else if (aktiF == SMAX) return "SMAX";
        else return "Unknown";
    }
};

class Neuron
{
public:

    Neuron(unsigned anzahl_an_neuronen_des_naechsten_Layers, unsigned my_Index, const LayerType &layerT, const double & init_range = 1.0);
    ~Neuron();

    //feed forward
    void aggregation(const Layer &prevLayer, const unsigned & neuron_count);
    void activation(const double &exp_sum = 0.0, const double &logsumexp = 0.0);
    // change weigts for genetic algorithm
    void mutate(const double &rate, const double & m_range);
    //get output
    double getOutputVal() const;

    //gradien calculation for feed forward back probagation
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer & nextLayer, unsigned neuroncount_with_bias);
    void updateInputWeights(Layer & prevLayer, const unsigned int &prevLayerNeuronCount, const double &eta, const double &alpha, const double clippingValue = 1.0);
    double sumDW(const Layer &nextLayer, unsigned neuroncount_with_bias) const;

    //intern
    unsigned int getConections_count() const;
    double getSumme_Aggregationsfunktion() const;
    char getType() const;

    void setOutputVal(const double &newOutputVal);
    double * m_outputWeights;
    double * delta_Weights;


    LayerType::Aggregationsfunktion getAggrF() const;
    LayerType::Aktivierungsfunktion getAktiF() const;

private:
    double activationFunction(const double &x, const double &exp_sum  = 0.0, const double &logsumexp = 0.0);
    double activationFunctionDerative(const double &x) const;

    double randomWeight();
    LayerType layerType;

    unsigned conections_count;
    double result_Aggregationsfunktion = 0;
    double m_outputVal = 0.0;
    unsigned my_Index;
    double m_gradient;
    double denominator, logsumexp;

    double softmax(const double &x) const;

};





#endif // NEURON_H
