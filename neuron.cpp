#include "neuron.h"
#include <cmath>
#include <iostream>
#include <ostream>


Neuron::Neuron(unsigned int anzahl_an_neuronen_des_naechsten_Layers, unsigned int my_Index, const LayerType &layerT)
    : layerType(layerT), conections_count(anzahl_an_neuronen_des_naechsten_Layers), my_Index(my_Index)
{
    m_outputWeights = new double[anzahl_an_neuronen_des_naechsten_Layers];
    for (unsigned c = 0; c < anzahl_an_neuronen_des_naechsten_Layers; ++c)  {
        m_outputWeights[c] = randomWeight();
    }
}

Neuron::~Neuron(){

    delete m_outputWeights;
}

std::mt19937/*minstd_rand*/ generator(std::random_device{}());
std::uniform_real_distribution<double> distribution(-1.0, 1.0);

double Neuron::randomWeight()
{
    return distribution(generator);
}

void Neuron::mutate(double rate)
{
    for(unsigned i = 0; i < conections_count; i++)
        if( std::abs(distribution(generator)) < rate) {
            m_outputWeights[i] += randomWeight() / 5.0;
            if(m_outputWeights[i] < -1)
                m_outputWeights[i] = -1;
            else if(m_outputWeights[i] > 1)
                m_outputWeights[i] = 1;
        }
}

void Neuron::activation(const double &exp_sum)
{
    m_outputVal = activationFunction(this->result_Aggregationsfunktion, exp_sum);
}


void Neuron::aggregation(const Layer &prevLayer, const unsigned int &neuron_count)
{
    this->result_Aggregationsfunktion = 0.0;

    switch (this->getAggrF()) {
    case LayerType::Aggregationsfunktion::SUM:
        for (unsigned n = 0; n < neuron_count; ++n)
            this->result_Aggregationsfunktion += prevLayer[n]->getOutputVal() * prevLayer[n]->m_outputWeights[my_Index];
        break;
    case LayerType::Aggregationsfunktion::AVG:
        for (unsigned n = 0; n < neuron_count; ++n)
            this->result_Aggregationsfunktion += prevLayer[n]->getOutputVal() * prevLayer[n]->m_outputWeights[my_Index];
        this->result_Aggregationsfunktion /= (double)neuron_count;
        break;
    case LayerType::Aggregationsfunktion::MAX:
        for (unsigned n = 0; n < neuron_count; ++n)
            this->result_Aggregationsfunktion = std::max(this->result_Aggregationsfunktion, prevLayer[n]->getOutputVal() * prevLayer[n]->m_outputWeights[my_Index]);
        break;
    case LayerType::Aggregationsfunktion::MIN:
        for (unsigned n = 0; n < neuron_count; ++n)
            this->result_Aggregationsfunktion = std::min(this->result_Aggregationsfunktion, prevLayer[n]->getOutputVal() * prevLayer[n]->m_outputWeights[my_Index]);
        break;
    }
}

double Neuron::activationFunction(const double &x, const double &exp_sum) const
{    
    switch (this->getAktiF()) {
    case LayerType::Aktivierungsfunktion::TANH:
        return std::tanh(x);
    case LayerType::Aktivierungsfunktion::RELU:
        return std::max(0.0, x);
    case LayerType::Aktivierungsfunktion::SMAX:
        return std::exp(x) / exp_sum;
    }
    return x;
}

LayerType::Aktivierungsfunktion Neuron::getAktiF() const
{
    return layerType.aktiF;
}

LayerType::Aggregationsfunktion Neuron::getAggrF() const
{
    return layerType.aggrF;
}


double Neuron::getOutputVal() const
{
    return m_outputVal;
}

void Neuron::setOutputVal(const double &newOutputVal)
{
    m_outputVal = newOutputVal;
}

unsigned int Neuron::getConections_count() const
{
    return conections_count;
}

double Neuron::getSumme_Aggregationsfunktion() const
{
    return result_Aggregationsfunktion;
}
