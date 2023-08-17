#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>

#include <random>

extern std::mt19937/*minstd_rand*/ generator;
extern std::uniform_real_distribution<double> distribution;
class Neuron;
typedef  Neuron** Layer;

enum Aggregationsfunktion  {
    SUM = 0,
    AVG,
    MAX,
    MIN,
};

enum Aktivierungsfunktion  {
    TANH = 0,
    RELU,
    SMAX
};


struct LayerType {
    LayerType()
    {

    }
    LayerType(    unsigned neuronCount,
              Aggregationsfunktion aggrF,
              Aktivierungsfunktion aktiF)
        : neuronCount(neuronCount), aggrF(aggrF), aktiF(aktiF)
    {

    }
    unsigned neuronCount;
    Aggregationsfunktion aggrF;
    Aktivierungsfunktion aktiF;
};

class Neuron
{
public:

    Neuron(unsigned anzahl_an_neuronen_des_naechsten_Layers, unsigned my_Index, const LayerType &layerT);
    ~Neuron();

    void aggregation(const Layer &prevLayer, const unsigned & neuron_count);
    void activation(const double &exp_sum = 0.0);
    void mutate(double rate);

    double getOutputVal() const;
    unsigned int getConections_count() const;
    double getSumme_Aggregationsfunktion() const;
    char getType() const;

    void setOutputVal(const double &newOutputVal);
    double * m_outputWeights;




    Aggregationsfunktion getAggrF() const;

    Aktivierungsfunktion getAktiF() const;

private:
    double activationFunction(const double &x, const double &exp_sum = 0) const;
    double randomWeight();
    Aggregationsfunktion aggrF;
    Aktivierungsfunktion aktiF;

    unsigned conections_count;
    double result_Aggregationsfunktion = 0;
    double m_outputVal = 0.0;
    unsigned my_Index;
};





#endif // NEURON_H
