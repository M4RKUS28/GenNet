#include "population.h"
#include <cstring>

Population::Population(const std::string &topology, const unsigned int &n_count, const double &init_range)
    : n_count(n_count), evolution(0), best(0)
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

void Population::evolve(const unsigned int &best, const double mutation_rate, const double mutation_range)
{
    if(best >= n_count) {
        perror("invalid number!");
        return;
    }

    this->evolution++;
    this->best = best;

    for(unsigned i = 0; i < n_count / 2; i++) {
        if(i == best)
            continue;
        if(nets[i]->createCopyFrom(nets[best]) == false )
            perror("Create copy failed failed!");
        nets[i]->mutate(mutation_rate, mutation_range);
    }

    //Hier aufteilen, einen Teil mut range und rate #ändern, einen teil mit zufälligen anderen muatationen evolven

    double part = (double) n_count / 4.0;
    std::mt19937/*minstd_rand*/ igenerator(std::random_device{}());
    std::uniform_int_distribution<int> idistribution(0, n_count - 1);

//    std::cout << "1: " << 0 << " bis excluse " << n_count / 2 << " 2: " << part * 2 << " bis excluse " << (unsigned)( part * 3) << " 3: " << part * 3 << " bis excluse " << n_count << std::endl;

    //startzufall weiter mutieren
    for(unsigned i = part * 2; i < (unsigned)( part * 3); i++) {
        int num = idistribution(igenerator);
        if(i == best || (unsigned)num == i)
            continue;
        if(!nets[i]->createCopyFrom(nets[num]))
            perror("Create copy failed failed!");
        nets[i]->mutate(0.01, 0.2);
    }

    //Hohe Mutation
    for(unsigned i = part * 3; i < n_count; i++) {
        if(i == best)
            continue;
        if(!nets[i]->createCopyFrom(nets[best]))
            perror("Create copy failed failed!");
        nets[i]->mutate(0.03, 0.3);
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

unsigned int Population::getBest() const
{
    return best;
}
