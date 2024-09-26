#include "agent.h"

Agent::Agent(std::string netTop, double gamma, double epsilon, unsigned int batchSize)
    :   Agent(new Net(netTop), gamma, epsilon, batchSize)
{

}

Agent::Agent(Net *net, double gamma, double epsilon, unsigned int batchSize)
{
    this->net = net;
    this->batchSize = batchSize;
    this->gamma = gamma;
    this->epsilon = epsilon;
}


void Agent::remember(const MEM_ENTRY &mem)
{
    memory.push_back(mem);
    //resize
    while(memory.size() > 0 && memory.size() > maxMemSize)
        memory.pop_front();
}

int argmax(const Action& vec) {
    if (vec.empty()) {
        // Handle the case where the vector is empty
        return -1; // or any other appropriate value
    }

    // Use std::max_element to find the iterator pointing to the maximum element
    auto maxElementIterator = std::max_element(vec.begin(), vec.end());

    // Check if maxElementIterator is valid before getting the index
    if (maxElementIterator != vec.end()) {
        // Calculate the index by subtracting the beginning iterator
        int index = std::distance(vec.begin(), maxElementIterator);
        return index;
    } else {
        // Handle the case where the vector is empty or other specific conditions
        return -1; // or any other appropriate value
    }
}


double *Agent::targetValue(const MEM_ENTRY &mem)
{
    int size = net->getTopology().back().neuronCount;
    double * retVal = new double[size];
    double * predic = new double[size];

    net->feedForward(mem.state.data());
    net->getResults(retVal);
    std::cout << " PREDICT: " << retVal[0] << " - " << retVal[1] << " - " << retVal[2] << " - " << retVal[3] << std::endl;

    double Q_new = mem.reward; // reward[idx]
    if(!mem.done) {
        net->feedForward(mem.nextState.data());
        net->getResults(predic);
        double max = *std::max_element(predic, predic + size);
        Q_new = mem.reward + this->gamma * max;
    }
    std::cout << " argmax(mem.action): " << argmax(mem.action) << std::endl;

    retVal[argmax(mem.action)] = Q_new;

    delete[] predic;
    std::cout << " TARGET: " << retVal[0] << " - " << retVal[1] << " - " << retVal[2] << " - " << retVal[3] << std::endl;

    return retVal;
}

void Agent::addStep(State lastState, State newState, Action action, double reward, bool done)
{
    MEM_ENTRY memEntr(lastState, action, reward, newState, done);
    this->train_short_memory(memEntr);
    this->remember(memEntr);
    if(done)
        this->train_long_memory();
}


Action Agent::getAction(const State &state)
{
    // random moves: tradeoff exploration / exploitation
    epsilon = 80 - /*n_games*/ 0;
    Action final_move;
    int size = net->getTopology().back().neuronCount;

    for(int i = 0; i < size; i++)
        final_move.push_back(0);

    if (std::rand() % 201 < epsilon) {
        int move = std::rand() % 3;
        final_move[move] = 1;
        std::cout << "GET RANDOM" << std::endl;
    } else {
        double* rawData = new double[size];
        net->feedForward(state.data());
        net->getResults(rawData);
        std::vector<double> prediction;
        for(int i = 0; i < size; i++)
            prediction.push_back(rawData[i]);
        delete[] rawData;

        int move = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        final_move[move] = 1;
    }

    std::cout << " GET ACTION: " << final_move[0] << " - " << final_move[1] << " - " << final_move[2] << " - " << final_move[3] << std::endl;

    return final_move;
}

void Agent::train_short_memory(const MEM_ENTRY &mem)
{
    double * targetVal = targetValue(mem);
    net->backProp(targetVal, 0.01, 0.15, false);
    // delete[] targetVal;
}


void Agent::train_long_memory()
{
    // Create a copy of the input deque
    std::deque<MEM_ENTRY> sampleDeque = memory;

    // Use a random device and a random number generator to generate random indices
    std::random_device rd;
    std::mt19937 gen(rd());

    // Iterate SIZE times and select random elements <<>> if there are still elements in the deque
    for (std::size_t i = 0; i < batchSize && sampleDeque.size(); ++i) {
        // Generate a random index within the remaining range of the deque
        std::size_t randomIndex = std::uniform_int_distribution<std::size_t>(0, sampleDeque.size() - 1)(gen);

        double * targetVal = targetValue(sampleDeque[randomIndex]);
        net->backProp(targetVal, 0.01, 0.15, true);
        // delete[] targetVal;

        // Remove the sampled element to avoid duplicate selections
        sampleDeque.erase(sampleDeque.begin() + randomIndex);
    }

    net->applyBatch();
    std::cout << "recentAverrageError(): " << net->recentAverrageError() << std::endl;

}
