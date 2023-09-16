#include "neuron.h"
#include <cmath>
#include <iostream>
#include <ostream>


Neuron::Neuron(unsigned int anzahl_an_neuronen_des_naechsten_Layers, unsigned int my_Index, const LayerType &layerT, const double &init_range)
    : layerType(layerT), conections_count(anzahl_an_neuronen_des_naechsten_Layers), my_Index(my_Index), m_gradient(0.0), denominator(0.0)

{
    m_outputWeights = new double[anzahl_an_neuronen_des_naechsten_Layers];
    delta_Weights = new double[anzahl_an_neuronen_des_naechsten_Layers];

    for (unsigned c = 0; c < anzahl_an_neuronen_des_naechsten_Layers; ++c)  {
        m_outputWeights[c] = randomWeight() * init_range;
        delta_Weights[c] = 0.0;

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



std::mt19937 gen(std::random_device{}());
std::normal_distribution<double> randomGaussianDistribution(0.0, 1.0);

void Neuron::mutate(const double &rate, const double &m_range)
{
    for(unsigned i = 0; i < conections_count; i++)
        if( std::abs(distribution(generator)) < rate) {
            m_outputWeights[i] += randomGaussianDistribution(gen) * m_range;
            if(m_outputWeights[i] < -1)
                m_outputWeights[i] = -1;
            else if(m_outputWeights[i] > 1)
                m_outputWeights[i] = 1;
        }
}

void Neuron::activation(const double &exp_sum, const double &logsumexp)
{
    m_outputVal = activationFunction(this->result_Aggregationsfunktion, exp_sum, logsumexp);
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

double Neuron::activationFunction(const double &x, const double &exp_sum, const double &logsumexp)
{
    //if(((x < 0.03 && exp_sum < 0.03) && this->getAktiF() == 2) || x < 0.003 )
    //    std::cout << "x: " << x << " exp_sum: " << exp_sum << " neuron: " << this->my_Index << " " << this->getAktiF() << std::endl;

    switch (this->getAktiF()) {
    case LayerType::Aktivierungsfunktion::TANH:
        return std::tanh(x);
    case LayerType::Aktivierungsfunktion::RELU:
        return std::max(0.0, x);
    case LayerType::Aktivierungsfunktion::SMAX:
        this->denominator = exp_sum;
        this->logsumexp = logsumexp;
        return softmax(x);
    }
    return  -1.0;
}

double Neuron::softmax(const double &x) const
{
    double val = std::exp(x - logsumexp);
    if(val == std::numeric_limits<double>::infinity() || val == std::numeric_limits<double>::signaling_NaN()) {
        perror("INVALID X VAL!");
        exit(12);
    }
    return val;
}

double Neuron::activationFunctionDerative(const double &x) const
{
    switch (this->getAktiF()) {
    case LayerType::Aktivierungsfunktion::TANH:
        return 1.0 - x*x;
    case LayerType::Aktivierungsfunktion::RELU:
        return (x > 0) ? 1.0 : 0.0;
    case LayerType::Aktivierungsfunktion::SMAX:
        double sigma_i = softmax(x);// /*softmax*/(std::exp(x) / denominator);
        return sigma_i * (1.0 - sigma_i) /** !ERRROR!   * denominator*/;
    }
    return x;
}



void Neuron::calcHiddenGradients(const Layer &nextLayer, unsigned int neuroncount_with_bias)
{
    double dow = sumDW(nextLayer, neuroncount_with_bias);
    m_gradient = dow * activationFunctionDerative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * activationFunctionDerative(m_outputVal);
}


void Neuron::updateInputWeights(Layer &prevLayer, const unsigned & prevLayerNeuronCount, const double & eta, const double & alpha, const double clippingValue)
{
    for (unsigned n = 0; n < prevLayerNeuronCount; ++n) {
        Neuron * neuron = prevLayer[n];

        double oldDeltaWeight = neuron->delta_Weights[my_Index];
        double newDeltaWeight = eta * neuron->getOutputVal() * m_gradient + alpha * oldDeltaWeight ; //<== HIER mit TZeit : 56:00 mint und https://www.youtube.com/watch?v=KkwX7FkLfug

        // Apply gradient clipping to newDeltaWeight
        if (newDeltaWeight > clippingValue) {
            perror("WARNING: CLIPPING TO 1");
            newDeltaWeight = clippingValue;
        } else if (newDeltaWeight < -clippingValue) {
            newDeltaWeight = -clippingValue;
            perror("WARNING: CLIPPING TO -1");
        }

        neuron->delta_Weights[(my_Index)] = newDeltaWeight;
        neuron->m_outputWeights[(my_Index)] += newDeltaWeight;

    }
}

double Neuron::sumDW(const Layer &nextLayer, unsigned neuroncount_with_bias) const
{
    double sum = 0.0;
    for (unsigned n = 0; n < neuroncount_with_bias; ++n)
    {
        sum += m_outputWeights[n] * nextLayer[n]->m_gradient;
    }
    return sum;
}


