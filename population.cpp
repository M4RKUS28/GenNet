#include "population.h"
#include <cstring>

Population::Population(const std::string &topology, const unsigned int &n_count, const double &init_range, const double init_t, const bool &useMutationThreads)
    : n_count(n_count), evolution(0), temp(init_t), useMutThreads(useMutationThreads)
{
    if(n_count <= 0) {
        perror("n_count < 0");
        exit(-5);
    }
    nets = new Net*[n_count];
    scores = new int[n_count];
    std::vector<LayerType > top = Net::getTopologyFromStr(topology);

    for(unsigned i = 0; i < n_count; i++) {
        nets[i] = new Net(top, init_range);
    }

}

int Population::indexOfBest() {
    int maxElement = scores[0];
    int index = 0;
    for (size_t i = 1; i < n_count; ++i) {
        if (scores[i] > maxElement) {
            maxElement = scores[i];
            index = i;
        }
    }
    return index;
}

Population::~Population()
{

    for(unsigned i = 0; i < n_count; i++) {
        delete nets[i];
    }
    delete[] nets;
    delete scores;
}

#include <atomic>
#include <list>
#include <unistd.h>
#include <list>


int Population::optimalThreadCount()
{
    return std::thread::hardware_concurrency() - ((std::thread::hardware_concurrency() > 1) ? 1 : 0);
}


void Population::evolve(const unsigned int &best, const double mutation_rate, const double mutation_range)
{
    if(best >= n_count || n_count <= 0) {
        perror("invalid input!");
        return;
    }

    std::vector<std::pair<std::thread, std::atomic<bool>*>> mutThreads;
    std::list<std::pair<int, int>> copyfrom_copyTo;
    unsigned processor_count = optimalThreadCount();

    /*FastRandom*/std::mt19937 igenerator(std::random_device{}());
    std::uniform_int_distribution<int> idistribution(0, n_count - 1);

    this->evolution++;

    double part = (double) n_count / 4.0;
    int optimum_task_split = n_count / (processor_count);
    if(optimum_task_split <= 0)
        optimum_task_split = 1;

    std::cout << "[0] --> < [" << (unsigned) (3.0 * part) << "]; [" << (unsigned) (part * 3.0) << "] --> < [ " << (unsigned)( part * 3.0 + part/2.0) << "]; [*] --> < [" << n_count << "]" << std::endl;


    for(unsigned i = 0; i < (unsigned) (3.0 * part); i++) {
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

    double diff_best_and_zero[3];
    if(n_count >= 3)     {
        diff_best_and_zero[0]   =  nets[       1    ]->getDifferenceFromOtherNet(nets[best]);
        diff_best_and_zero[1]   =  nets[n_count -1  ]->getDifferenceFromOtherNet(nets[best]);
        diff_best_and_zero[2]   =  nets[n_count /2+1]->getDifferenceFromOtherNet(nets[best]);

    }

    if(useMutThreads)
        for(unsigned i = 0; i < mutThreads.size(); i++) {
            if(mutThreads.at(i).first.joinable()) {
                mutThreads.at(i).first.join();
                delete mutThreads.at(i).second;
            }
        }
    if(n_count >= 3)     {
        std::cout << "Evo: " << evolution << " Average mutation: [1] is "
              << diff_best_and_zero[0] << ", [" << n_count -1 << "] is "
              << diff_best_and_zero[1] << ", [" << n_count /2+1 << "] is "
              << diff_best_and_zero[2] << " differned from best snake [" << best << "] " << std::endl /*<< "INP: "*/;
    }
    else
        std::cout << "Evo: " << evolution << std::endl;
}

int *Population::scoreMap()
{
    return scores;
}



void Population::evolveWithSimulatedAnnealing(const double mutation_rate, const double mutation_range, const double &temp_rate)
{
    if(n_count <= 0 || temp == 0) {
        return perror("invalid input!");
    }

    std::vector<std::pair<std::thread, std::atomic<bool>*>> mutThreads;
    std::list<std::pair<int, int>> copyfrom_copyTo;
    unsigned processor_count = optimalThreadCount();

    std::mt19937 igenerator(std::random_device{}());
    std::uniform_int_distribution<int> idistribution(0, n_count - 1);
    std::uniform_real_distribution<double> prop_dis(0.0, 1.0);  // uniform distribution [0, 1)

    int optimum_task_split = ((n_count / (processor_count)) <= 0) ? (1) : (n_count / (processor_count));

    this->evolution++;
    this->temp = this->temp * temp_rate;
    //std::swap(nets[0], nets[indexOfBest()]); // set index of best to 0
    unsigned best = indexOfBest();
    int last_best_mutated = -1;
    int worseOneGenommen = 0;

    for(unsigned i = 0; i < n_count; i++) {
        if(i != best) {
            // if(i==0)
            //     std::cout << "i=0: scores[i] >= scores[best]: " << scores[i]<< "  >=  " <<  scores[best]  << std::endl;

            if(scores[i] >= scores[best]) {
                // net is better or as good as best -> use it to mutate
                copyfrom_copyTo.push_back(std::pair<int, int>(i, i));

            } else if(scores[best]) {
                double probability = std::exp( ( (scores[i] / scores[best]) - 1.0 ) / this->temp); //std::exp(- (( scores[best] - scores[i]) / this->temp));
                // if(i==-1)
                //     std::cout << "i=0: prob: " << prop_dis(igenerator) << " <?< " <<  probability << std::endl;
                bool useWorseOne = prop_dis(igenerator) < probability;
                if(useWorseOne) {
                    copyfrom_copyTo.push_back(std::pair<int, int>(i, i));
                    last_best_mutated = i;
                    worseOneGenommen++;
                } else {
                    copyfrom_copyTo.push_back(std::pair<int, int>(best, i));
                }
            }
        }
        //start part
        if(i % optimum_task_split == 0 || i == n_count - 1) {
            do_create_new(&mutThreads, copyfrom_copyTo, processor_count, nets, mutation_rate * temp, mutation_range );
            copyfrom_copyTo.clear();
        }
    }


    if(useMutThreads)
        for(unsigned i = 0; i < mutThreads.size(); i++) {
            if(mutThreads.at(i).first.joinable()) {
                mutThreads.at(i).first.join();
                delete mutThreads.at(i).second;
            }
        }

    if(last_best_mutated >= 0 && last_best_mutated < (int)n_count)     {
        std::cout << " Average mutation: [best] to [mutated_best] is "
                  << nets[last_best_mutated]->getDifferenceFromOtherNet(nets[best])
                  << std::endl;
    }
    std::cout << " Temperature: " << this->temp << " | Nimmt Netz zu 50% NICHT! wenn Score " << this->temp *( std::log(0.5) + 1.0) << " % von MaxScore = "  << scores[best]
              << ", also " << (this->temp * (std::log(0.5) + 1.0)) * (double) scores[best] << " erreicht!" /*temp*std::log(0.5)*/ << std::endl;
    std::cout << " worseOneGenommen-Percentage: " << (int)(100.0 * (double)worseOneGenommen / (double)n_count) << " % --> abs: " << worseOneGenommen << std::endl;
    std::cout << " current mut rate: " << temp * mutation_rate << std::endl;


    std::cout << "Finished Evolution: " << evolution << std::endl;
}

double Population::getTemperature()
{
    return temp;
}

void Population::setTemperature(const double &t)
{
    temp = t;
}

unsigned int Population::getEvolutionNum() const
{
    return evolution;
}

Net *Population::netAt(const unsigned int &index)
{
    return nets[index];
}

// unsigned int Population::getBest() const
// {
//     return best;
// }

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
        mutThreads->push_back(std::pair<std::thread, std::atomic<bool>* >(std::thread(this->mutateThread, nets, copyfrom_copyTo, mut_rate, range, isFinished), isFinished));
    } else {
        mutateThread(nets, copyfrom_copyTo, mut_rate, range);
    }
}

void Population::mutateThread(Net **nets, const std::list<std::pair<int, int> > copyfrom_copyTo, const double mut_rate, const double range, std::atomic_bool *is_finished)
{
    for(auto it = copyfrom_copyTo.begin(); it != copyfrom_copyTo.end(); ++it) {
        if(!nets[it->second]->createCopyFrom(nets[it->first]))
            perror("Create copy failed failed!");
        else
            nets[it->second]->mutate(mut_rate, range);
        // std::cout << " COPIED: net[" << it->first << "] to net[" << it->second << "] and mutated [" << it->second << "]" << std::endl;
    }
    if(is_finished)
        *is_finished = true;
}
