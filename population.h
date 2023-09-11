#ifndef Population_H
#define Population_H

#include "net.h"




class Population
{
public:
    Population(const std::string &topology, const unsigned &size);
    ~Population();

    void evolve(const unsigned & best, const double mutation_rate, const double mutation_range = 1.0);
    unsigned int getEvolutionNum() const;

    Net * netAt(const unsigned &index);
    Net ** nets;

    unsigned int getBest() const;

private:
    unsigned n_count;
    unsigned evolution;
    unsigned best;
};

#endif // Population_H
