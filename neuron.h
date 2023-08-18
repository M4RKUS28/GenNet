#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>

#include <random>

extern std::mt19937/*minstd_rand*/ generator;
extern std::uniform_real_distribution<double> distribution;
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




    LayerType::Aggregationsfunktion getAggrF() const;

    LayerType::Aktivierungsfunktion getAktiF() const;

private:
    double activationFunction(const double &x, const double &exp_sum = 0) const;
    double randomWeight();
    LayerType layerType;

    unsigned conections_count;
    double result_Aggregationsfunktion = 0;
    double m_outputVal = 0.0;
    unsigned my_Index;
};





#endif // NEURON_H
