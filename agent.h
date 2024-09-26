#ifndef AGENT_H
#define AGENT_H

#include "net.h"
#include <deque>
#include <vector>
#include <algorithm>


typedef std::vector<double> State;
typedef std::vector<int> Action;

class Agent
{
public:
    Agent(std::string netTop, double gamma = 0.9, double epsilon = 0.0, unsigned batchSize = 1000);
    Agent(Net * net, double gamma = 0.9, double epsilon = 0.0, unsigned batchSize = 1000);


    struct MEM_ENTRY {
        MEM_ENTRY() {

        }
        MEM_ENTRY(State state, Action action,  double reward, State nextState,    bool done )
            : state(state),  action(action)  ,reward(reward), nextState(nextState)  ,done(done)
        {

        }

        State state;
        Action action;
        double reward;
        State nextState;
        bool done;
    };


    void train_short_memory(const MEM_ENTRY &mem);
    void train_long_memory();
    void remember(const MEM_ENTRY &mem);

    double * targetValue(const MEM_ENTRY &mem);

public:
    void addStep(State lastState, State newState, Action action, double reward, bool done);
    Action getAction(const State &state);

    //public virtual...
    std::deque<MEM_ENTRY> memory;
    Net * net;
    const unsigned maxMemSize = 100000;
    unsigned batchSize;
    double gamma;
    double epsilon;

};

#endif // AGENT_H
