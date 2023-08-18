#ifndef NET_H
#define NET_H

#include "neuron.h"

#include <vector>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>


typedef Neuron** Layer;

class Net
{
public:
    Net(const std::string &topology);
    Net(const std::vector<LayerType> &topology);
    Net(const Net *other);
    Net(const std::string filename, bool &ok);
    ~Net();

    bool save_to(const std::string &path);
    bool load_from(const std::string &path);


    static std::vector<LayerType> getTopologyFromStr(const std::string & top);
    std::string getTopologyStr();

    void feedForward(const double *input);
    void getResults(double *output) const;

    bool createCopyFrom(const Net *origin);
    std::vector<LayerType> getTopology() const;

    void mutate(const double & mutation_rate);

    double getConWeight(const unsigned int &layer, const unsigned int &neuronFrom, const unsigned int &neuronTo) const;
    double getNeuronValue(const unsigned int &layer, const unsigned int &neuron) const;

private:
    unsigned layerCount() const;
    unsigned neuronCountAt(const unsigned &layer) const;

    Layer* m_layers;
    std::vector<LayerType> topology;

protected:
    void init();
};




#endif // NET_H
