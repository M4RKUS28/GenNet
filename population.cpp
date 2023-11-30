#include "population.h"
#include <cstring>

Population::Population(const std::string &topology, const unsigned int &n_count, const double &init_range, const bool &useMutationThreads)
    : n_count(n_count), evolution(0), best(0), useMutThreads(useMutationThreads)
{
    nets = new Net*[n_count];
    std::vector<LayerType > top = Net::getTopologyFromStr(topology);

    for(unsigned i = 0; i < n_count; i++) {
        nets[i] = new Net(top, init_range);
    }

}

Population::~Population()
{

    for(unsigned i = 0; i < n_count; i++) {
        delete nets[i];
    }
    delete[] nets;
}

#include <atomic>
#include <list>
#include <unistd.h>
#include <list>


void Population::evolve(const unsigned int &best, const double mutation_rate, const double mutation_range)
{
    if(best >= n_count) {
        perror("invalid number!");
        return;
    }

    std::vector<std::pair<std::thread, std::atomic<bool>*>> mutThreads;
    unsigned processor_count = std::thread::hardware_concurrency() - ((std::thread::hardware_concurrency() > 2) ? 2 : 0);
    std::list<std::pair<int, int>> copyfrom_copyTo;

    FastRandom/*std::mt19937*//*minstd_rand*/ igenerator(std::random_device{}());
    std::uniform_int_distribution<int> idistribution(0, n_count - 1);

    this->evolution++;
    this->best = best;

    double part = (double) n_count / 4.0;
    int optimum_task_split = n_count / (processor_count);
    std::cout << "[0] --> < [" << (unsigned) 3.0 * part << "]; [" << (unsigned) part * 3.0 << "] --> < [ " << (unsigned)( part * 3.0 + part/2.0) << "]; [*] --> < [" << n_count << "]" << std::endl;


    for(unsigned i = 0; i < (unsigned) 3.0 * part; i++) {
        if(i == best)
            continue;
        else if(i % optimum_task_split == 0) {
            do_create_new(&mutThreads, copyfrom_copyTo, processor_count, nets, mutation_rate, mutation_range );
            copyfrom_copyTo.clear();
        } else
            copyfrom_copyTo.push_back(std::pair<int, int>(best, i));
    }

    //Hier aufteilen, einen Teil mut range und rate #ändern, einen teil mit zufälligen anderen muatationen evolven
    //startzufall weiter mutieren
    for(unsigned i = part * 3.0; i < (unsigned)( part * 3.0 + part/2.0); i++) {
        int num = idistribution(igenerator);
        if(i == best || (unsigned)num == i)
            continue;
        else if(i % optimum_task_split == 0) {
            do_create_new(&mutThreads, copyfrom_copyTo, processor_count, nets, 0.01, 0.2 );
            copyfrom_copyTo.clear();
        } else
            copyfrom_copyTo.push_back(std::pair<int, int>(best, i));
    }

    //Hohe Mutation
    for(unsigned i = (unsigned)( part * 3.0 + part/2.0); i < n_count; i++) {
        if(i == best)
            continue;
       else if(i % optimum_task_split == 0) {
            do_create_new(&mutThreads, copyfrom_copyTo, processor_count, nets, 0.03, 0.3);
            copyfrom_copyTo.clear();
       } else
                copyfrom_copyTo.push_back(std::pair<int, int>(best, i));
    }

    // do rest...
    do_create_new(&mutThreads, copyfrom_copyTo, processor_count, nets, 0.01, 0.2 );
    copyfrom_copyTo.clear();

    double diff_best_and_zero[3] = { nets[       1    ]->getDifferenceFromOtherNet(nets[best]),
                                     nets[n_count -1  ]->getDifferenceFromOtherNet(nets[best]),
                                     nets[n_count /2+1]->getDifferenceFromOtherNet(nets[best])};

    if(useMutThreads)
        for(unsigned i = 0; i < mutThreads.size(); i++) {
            if(mutThreads.at(i).first.joinable()) {
                mutThreads.at(i).first.join();
                delete mutThreads.at(i).second;
            }
        }

    std::cout << "Evo: " << evolution << " Average mutation: [1] is "
              << diff_best_and_zero[0] << ", [" << n_count -1 << "] is "
              << diff_best_and_zero[1] << ", [" << n_count /2+1 << "] is "
              << diff_best_and_zero[2] << " differned from best snake [" << best << "] " << std::endl /*<< "INP: "*/;
}

unsigned int Population::getEvolutionNum() const
{
    return evolution;
}

Net *Population::netAt(const unsigned int &index)
{
    return nets[index];
}

unsigned int Population::getBest() const
{
    return best;
}

void Population::do_create_new(std::vector<std::pair<std::thread, std::atomic<bool> *> > *mutThreads, const std::list<std::pair<int, int>> copyfrom_copyTo, const unsigned int thread_max_count, Net **nets, const double &mut_rate, const double &range)
{
    if(useMutThreads) {
        std::cout << "[ " << copyfrom_copyTo.back().second << " / " << n_count <<" ]" << mutThreads->size() << " / " << thread_max_count << " Threads: waiting until next finishes..." << std::endl;

        while(mutThreads->size() >= thread_max_count) {
            for(unsigned i = 0; i < mutThreads->size(); i++) {
                if( * mutThreads->at(i).second) {
                    if(mutThreads->at(i).first.joinable()) {
                        mutThreads->at(i).first.join();
                        //delete atomic var
                        delete mutThreads->at(i).second;
                        mutThreads->erase(mutThreads->begin() + i);
                        i--;
                    }
                }
            }
            if(mutThreads->size() < thread_max_count)
                break;
            else {
                //std::cout << "usleep(50000)" << std::endl;
                usleep(50000);
            }
        }
        std::atomic<bool> *isFinished = new std::atomic<bool>(false);
        mutThreads->push_back(std::pair<std::thread, std::atomic<bool>* >(std::thread(this->createNetThread, nets, copyfrom_copyTo, mut_rate, range, isFinished), isFinished));
    } else {
        createNetThread(nets, copyfrom_copyTo, mut_rate, range);
    }
}

void Population::createNetThread(Net **nets, const std::list<std::pair<int, int> > copyfrom_copyTo, const double mut_rate, const double range, std::atomic_bool *is_finished)
{
    for(auto it = copyfrom_copyTo.begin(); it != copyfrom_copyTo.end(); ++it) {
        if(!nets[it->second]->createCopyFrom(nets[it->first]))
            perror("Create copy failed failed!");
        else
            nets[it->second]->mutate(mut_rate, range);
    }
    if(is_finished)
        *is_finished = true;
}
