#ifndef Population_H
#define Population_H

#include "net.h"




class Population
{
public:
    Population(const std::string &topology, const unsigned &size);
    void evolve(const unsigned & best, const double mutation_rate);
    unsigned int getEvolutionNum() const;

    Net * netAt(const unsigned &index);
    Net ** nets;

private:
    unsigned n_count;
    unsigned evolution;
    unsigned best;
};

#endif // Population_H
