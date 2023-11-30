#ifndef Population_H
#define Population_H

#include "net.h"


#include <thread>
#include <atomic>
#include <list>

class Population
{
public:

    Population(const std::string &topology, const unsigned &size, const double & init_range = 0.1, const bool & useMutationThreads = false);

    ~Population();

    void evolve(const unsigned & best, const double mutation_rate, const double mutation_range = 1.0);
    unsigned int getEvolutionNum() const;

    Net * netAt(const unsigned &index);
    Net ** nets;

    unsigned int getBest() const;

private:

    void do_create_new(std::vector<std::pair<std::thread, std::atomic<bool> *>> *muThreads, const std::list<std::pair<int, int>> copyfrom_copyTo, const unsigned thread_max_count, Net **nets, const double & mut_rate, const double &range);
    static void createNetThread(Net **nets, const std::list<std::pair<int, int>> copyfrom_copyTo, const double mut_rate, const double range, std::atomic_bool *is_finished  = nullptr);
    unsigned n_count;
    unsigned evolution;
    unsigned best;
    bool useMutThreads;
};

#endif // Population_H
