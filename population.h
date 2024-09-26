#ifndef Population_H
#define Population_H

#include "net.h"


#include <thread>
#include <atomic>
#include <list>

class Population
{
public:

    Population(const std::string &topology, const unsigned &size, const double & init_range = 0.1, const double init_t = 1.0, const bool & useMutationThreads = false);

    ~Population();

    void evolve(const unsigned & best, const double mutation_rate, const double mutation_range = 0.2);

    int * scoreMap();
    void evolveWithSimulatedAnnealing(const double mutation_rate, const double mutation_range = 0.2, const double &tempRate = 0.9);
    double getTemperature();
    void setTemperature(const double &t);

    unsigned int getEvolutionNum() const;


    Net * netAt(const unsigned &index);
    Net ** nets;

private:
    void do_create_new(std::vector<std::pair<std::thread, std::atomic<bool> *>> *muThreads, const std::list<std::pair<int, int>> copyfrom_copyTo, const unsigned thread_max_count, Net **nets, const double & mut_rate, const double &range);
    static void mutateThread(Net **nets, const std::list<std::pair<int, int>> copyfrom_copyTo, const double mut_rate, const double range, std::atomic_bool *is_finished  = nullptr);
    static int optimalThreadCount();
    int indexOfBest();

    unsigned n_count;
    unsigned evolution;
    double temp;

    bool useMutThreads;
    int * scores;
};

#endif // Population_H
