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
    Net(const std::string &topology, const double &init_range = 1.0);
    Net(const std::vector<LayerType> &topology, const double &init_range = 1.0);
    Net(const Net *other);
    Net(const std::string filename, bool &ok);
    ~Net();

    //load store net -> iofstreams
    bool save_to(const std::string &path);
    bool load_from(const std::string &path);

    //topology stuff
    std::vector<LayerType> getTopology() const;
    static std::vector<LayerType> getTopologyFromStr(const std::string & top);
    std::string getTopologyStr();

    //net main functions:
    //<<In
    void feedForward(const double *input);
    //>>out
    void getResults(double *output) const;
    //~optimizer
    void backProp(double * targetVals, const double &eta, const double &alpha, const double &range_max = 1.0);
    void mutate(const double & mutation_rate, const double & mutation_range = 1.0 );

    //make copy from other
    bool createCopyFrom(const Net *origin);

    //get infos
    double getConWeight(const unsigned int &layer, const unsigned int &neuronFrom, const unsigned int &neuronTo) const;
    double getNeuronValue(const unsigned int &layer, const unsigned int &neuron) const;

    double recentAverrageError() const;

private:
    unsigned layerCount() const;
    unsigned neuronCountAt(const unsigned &layer) const;

    Layer* m_layers;
    std::vector<LayerType> topology;

    double m_error;
    double m_recentAverangeSmoothingFactor;
    double m_recentAverrageError;

protected:
    void init(const double &init_range = 1.0);
};




#endif // NET_H
