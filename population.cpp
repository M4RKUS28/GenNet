#include "population.h"
#include <cstring>

Population::Population(const std::string &topology, const unsigned int &n_count)
    : n_count(n_count), evolution(0), best(0)
{
    nets = new Net*[n_count];
    std::vector<LayerType > top = Net::getTopologyFromStr(topology);

    for(unsigned i = 0; i < n_count; i++) {
        nets[i] = new Net(top);
    }

}

void Population::evolve(const unsigned int &best, const double mutation_rate)
{
    if(best >= n_count) {
        perror("invalid number!");
        return;
    }

    this->evolution++;
    this->best = best;

    for(unsigned i = 0; i < n_count; i++) {
        if(i == best)
            continue;
        delete nets[i];
        nets[i] = new Net(nets[best]);
        nets[i]->mutate(mutation_rate);
    }

    std::cout << "Evo: " << evolution << std::endl /*<< "INP: "*/;
}

unsigned int Population::getEvolutionNum() const
{
    return evolution;
}

Net *Population::netAt(const unsigned int &index)
{
    return nets[index];
}
