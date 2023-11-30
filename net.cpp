#include "net.h"


Net::Net(const std::vector<LayerType> &topology, const double &init_range, const bool &enable_batch_learning)
    :  topology(topology)
{
    // create Net
    init(init_range, enable_batch_learning);
}

Net::Net(const std::string &s_topology, const double &init_range, const bool &enable_batch_learning)
    :  topology(getTopologyFromStr(s_topology))
{
    // create Net
    init(init_range, enable_batch_learning);
}


Net::Net(const Net *other, const double &init_range, const bool &enable_batch_learning)
    : Net(other->getTopology())
{
    // create Net
    init(init_range, enable_batch_learning);
    // copy Net
    createCopyFrom(other);
}

Net::Net(const std::string filename, bool &ok, const bool &enable_batch_learning)
{
    ok = load_from(filename, enable_batch_learning);
}


void Net::init(const double &init_range, const bool & enable_batch_learning)
{
    if(topology.size() < 2) {
        std::cerr << "Invalid net: Topology-size is to small or 0: " << topology.size() << std::endl;
        return;
    }
    // Create Net
    m_layers = new Layer[topology.size()];

    for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
        unsigned neuron_count = topology.at(layerNum).neuronCount + 1 /*BIAS*/;

        m_layers[layerNum] = new Neuron*[neuron_count];
        // speichere Nummer des Nächsten Layers, esseiden, dieser ist der Letzte, dann speichere 0
        unsigned anzahl_an_neuronen_des_naechsten_Layers = (layerNum == topology.size() - 1) ? 0 : topology.at(layerNum + 1).neuronCount;

        for (unsigned neuronNum = 0; neuronNum < neuron_count; ++neuronNum)  {
            m_layers[layerNum][neuronNum] = new Neuron(anzahl_an_neuronen_des_naechsten_Layers, neuronNum, topology.at(layerNum), init_range, enable_batch_learning);
        }

        /*BIAS*/
        m_layers[layerNum][topology.at(layerNum).neuronCount]->setOutputVal(1);
    }


    m_recentAverangeSmoothingFactor = 0.0009;
    m_recentAverrageError = 0.0;
}
Net::~Net() {

    for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
        for (unsigned neuronNum = 0; neuronNum < topology.at(layerNum).neuronCount + 1 /*BIAS*/; ++neuronNum) {
            delete m_layers[layerNum][neuronNum];
        }
        delete m_layers[layerNum];
    }
    delete m_layers;
}


bool Net::save_to(const std::string &filename)
{
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {

        //1. Line Topology!
        outputFile << getTopologyStr() << "\n------\n";

        for (unsigned layer = 0; layer < layerCount() - 1 /*letzte hat keine verbindungen*/; ++layer) {
            for (unsigned neuron = 0; neuron < neuronCountAt(layer) + 1/*BIAS*/; ++neuron) {
                for (unsigned weight = 0; weight < m_layers[layer][neuron]->getConections_count(); ++weight) {
                    outputFile << m_layers[layer][neuron]->m_outputWeights[weight];
                    if (weight != m_layers[layer][neuron]->getConections_count() - 1) {
                        outputFile << ",";
                    }
                }
                outputFile << "\n";
            }
            outputFile << "\n";
        }
        outputFile.close();
        std::cout << "Neural network saved to " << filename << std::endl;
        return true;
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
        return false;
    }
}


bool Net::load_from(const std::string &filename, const bool &enable_batch_learning)
{
    //layerCount() // 4
    std::ifstream inputFile(filename);
    if (inputFile.is_open()) {
        std::string top_lines[2];
        if (!std::getline(inputFile, top_lines[0]) || !std::getline(inputFile, top_lines[1])) {
            perror("Error: reading line");
            return false;
        } else if(top_lines[0].empty()) {
            std::cerr << "No Topology! Invalid File!" << std::endl;
            return false;
        }

        //Init if necessary!
        if(getTopology().empty() == false) {
            if(getTopologyStr() != top_lines[0] ) {
                std::cerr << "Net already initialised with wrong topology! Init(" << getTopologyStr() << ") Given: (" << top_lines[0] << ")!" << std::endl;
                return false;
            } else {
                //Init...
                this->topology = getTopologyFromStr(top_lines[0]);
                if(topology.size() == 0 ) {
                    std::cerr << "Invalid Topology! load_from File failed!" << std::endl;
                    return false;
                }
                // Create Neurons....
                init(0.0, enable_batch_learning);

                // Load Weights.... ->
            }
        }

        //Load weights...
        for (unsigned layer = 0; layer < layerCount() - 1 /*letzter Layer braucht keine Gewichte!!!!*/; ++layer) {
            for (unsigned neuron = 0; neuron < neuronCountAt(layer) + 1/*BIAS!!!!!!*/; ++neuron) {
                std::string line;
                if (!std::getline(inputFile, line)) {
                    // Error reading line
                    perror("Error: reading line.");
                    return false;
                }
                std::vector<double> weights;
                std::istringstream iss(line);
                double value;
                while (iss >> value) {
                    weights.push_back(value);
                    if (iss.peek() == ',')
                        iss.ignore();
                }

                if (weights.size() != topology.at(layer + 1).neuronCount) {
                    // Error: Mismatch in weight count
                    std::cerr << "Error: Mismatch in weight count: " << weights.size() << " vs " << topology.at(layer + 1).neuronCount << std::endl;
                    return false;
                }

                for (unsigned weightIdx = 0; weightIdx < weights.size(); ++weightIdx) {
                    m_layers[layer][neuron]->m_outputWeights[weightIdx] = weights[weightIdx];
                }
            }

            //Leerzeile entfernen!
            std::string line;
            if (!std::getline(inputFile, line)) {
                // Error reading line
                perror("Error: reading line..");
                return false;
            }

        }
        inputFile.close();
        std::cout << "Neural network loaded from " << filename << std::endl;
        return true;
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
        return false;
    }
}

std::vector<LayerType> Net::getTopologyFromStr(const std::string &top)
{
    std::istringstream stream(top);
    std::string substring;
    std::vector<LayerType> retTop;

    while (std::getline(stream, substring, ',')) {
        int count = 0;
        size_t start = -1;
        LayerType lt;
        substring += '_';

        for(size_t index = substring.find('_'); index != std::string::npos; index = substring.find('_', start + 1)) {
            std::string part_s = substring.substr(start + 1, index - start - 1);
            if(part_s.length() == 0) {
                std::cerr << "Invalid argument: " << substring << std::endl;
                break;
            }

            if(count == 0) {// Topology
                try {
                    lt.neuronCount = std::abs(std::stoi(part_s));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid argument: " << e.what() << std::endl;
                    break;
                } catch (const std::out_of_range& e) {
                    std::cerr << "Out of range: " << e.what() << std::endl;
                    break;
                }

                //Aggregationsfunktion
            } else if(count == 1) {

                if(part_s != "INPUT" && part_s.length() != 3 ) {
                    std::cerr << "Invalid length Aggregationsfunktion : " << part_s << std::endl;
                    break;
                } else {
                    if(part_s == "SUM")
                        lt.aggrF = LayerType::Aggregationsfunktion::SUM;
                    else if(part_s == "MAX")
                        lt.aggrF = LayerType::Aggregationsfunktion::MAX;
                    else if(part_s == "MIN")
                        lt.aggrF = LayerType::Aggregationsfunktion::MIN;
                    else if(part_s == "AVG")
                        lt.aggrF = LayerType::Aggregationsfunktion::AVG;
                    else if(part_s == "INPUT")
                        lt.aggrF = LayerType::Aggregationsfunktion::INPUT_LAYER;
                    else {
                        std::cerr << "Invalid Aggregationsfunktion: " << part_s << std::endl;
                        break;
                    }
                }

                // Aktivierungsfunktion
            } else if(count == 2) {

                if((part_s != "LAYER" && part_s != "IDENTITY" && part_s != "SOFTPLUS" && part_s != "LEAKYRELU" && part_s != "SIGMOID") && part_s.length() != 4) {
                    std::cerr << "Invalid Aktivierungsfunktion: " << part_s << std::endl;
                    break;
                }else {
                    if(part_s == "TANH")
                        lt.aktiF = LayerType::Aktivierungsfunktion::TANH;
                    else if(part_s == "RELU")
                        lt.aktiF = LayerType::Aktivierungsfunktion::RELU;
                    else if(part_s == "SMAX")
                        lt.aktiF = LayerType::Aktivierungsfunktion::SMAX;
                    else if(part_s == "LAYER") /*INPUT_LAYER*/
                        lt.aktiF = LayerType::Aktivierungsfunktion::NONE;
                    else if(part_s == "IDENTITY") /*INPUT_LAYER*/
                        lt.aktiF = LayerType::Aktivierungsfunktion::IDENTITY;
                    else if(part_s == "SOFTPLUS") /*INPUT_LAYER*/
                        lt.aktiF = LayerType::Aktivierungsfunktion::SOFTPLUS;
                    else if(part_s == "LEAKYRELU") /*INPUT_LAYER*/
                        lt.aktiF = LayerType::Aktivierungsfunktion::LEAKYRELU;
                    else if(part_s == "SIGMOID") /*INPUT_LAYER*/
                        lt.aktiF = LayerType::Aktivierungsfunktion::SIGMOID;
                    else {
                        std::cerr << "Invalid Aktivierungsfunktion: " << part_s << std::endl;
                        break;
                    }
                }

                // Rest...
            } else {
                std::cerr << "Invalid Input - More than 2 x '-': " << substring << std::endl;
                break;
            }

            //WENN ALLES OK Zähl 1 hoch. Vorraussetzung: count == 3
            count++;
            start = index;
        }

        if(count != 3) {
            std::cerr << "Ungültige Eingabe: " << substring << std::endl;
            continue;
        } else {
            retTop.push_back(lt);
        }
    }

    return retTop;
}

std::string Net::getTopologyStr() const
{
    std::string retV;
    for (unsigned i = 0; i < getTopology().size(); ++i)
        retV += std::to_string(getTopology().at(i).neuronCount) + "_" + getTopology().at(i).aggregationsfunktionToString() + "_" + getTopology().at(i).aktivierungsfunktionToString() + ((i != getTopology().size() - 1) ? "," : "");
    return retV;
}


bool Net::createCopyFrom(const Net *origin)
{
    if(topology.size() != origin->topology.size()) {
        std::cerr << "topology.size() != origin->topology.size(): " << topology.size() << " != " << origin->topology.size() << std::endl;
        return false;
    }

    for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {

        if(topology.at(layerNum).neuronCount != origin->topology.at(layerNum).neuronCount)   {
            std::cerr << "topology.at(layerNum).neuronCount != origin->topology.at(layerNum).neuronCount: " << topology.at(layerNum).neuronCount << " != " << origin->topology.at(layerNum).neuronCount << std::endl;
            return false;
        }

        for (unsigned neuronNum = 0; neuronNum < topology.at(layerNum).neuronCount + 1 /*bias*/; ++neuronNum) {

            if(m_layers[layerNum][neuronNum]->getConections_count() != origin->m_layers[layerNum][neuronNum]->getConections_count())  {
                std::cerr << "m_layers[layerNum][neuronNum]->getConections_count() != origin->m_layers[layerNum][neuronNum]->getConections_count(): "
                          << m_layers[layerNum][neuronNum]->getConections_count() << " != " << origin->m_layers[layerNum][neuronNum]->getConections_count() << std::endl;
                return false;
            }
            for(unsigned c = 0; c < this->m_layers[layerNum][neuronNum]->getConections_count(); c++) {
                this->m_layers[layerNum][neuronNum]->m_outputWeights[c] = origin->m_layers[layerNum][neuronNum]->m_outputWeights[c];
            }
        }
    }
    return true;
}

double Net::getDifferenceFromOtherNet(const Net *other)
{
    if(other->getTopologyStr() != getTopologyStr()) {
        std::cerr << "Got net with differned top for difference!" << std::endl;
        return -1;
    }
    double diff = 0.0;
    size_t w_count = 0;

    for (unsigned layerNum = 0; layerNum < topology.size(); ++layerNum) {
        for (unsigned neuronNum = 0; neuronNum < topology.at(layerNum).neuronCount + 1 /*bias*/; ++neuronNum) {
            for(unsigned c = 0; c < this->m_layers[layerNum][neuronNum]->getConections_count(); c++) {
                diff += abs( (this->m_layers[layerNum][neuronNum]->m_outputWeights[c] - other->m_layers[layerNum][neuronNum]->m_outputWeights[c]));
                w_count ++;
            }
        }
    }

    return diff / (double)w_count;
}



std::vector<LayerType> Net::getTopology() const
{
    return topology;
}


void Net::mutate(const double &mutation_rate, const double &mutation_range)
{
    for (unsigned layerNum = 0; layerNum < topology.size() - 1/**???*/; ++layerNum) {
        for (unsigned neuronNum = 0; neuronNum < topology.at(layerNum).neuronCount + 1 /*bias*/; ++neuronNum) {
            for(unsigned c = 0; c < this->m_layers[layerNum][neuronNum]->getConections_count(); c ++) {
                this->m_layers[layerNum][neuronNum]->mutate(mutation_rate, mutation_range);
            }
        }
    }
}

double Net::getConWeight(const unsigned int & layer, const unsigned int &neuronFrom, const unsigned int& neuronTo) const
{
    if(layer >= layerCount() || neuronFrom >= neuronCountAt(layer) + 1 /*bias*/ || neuronTo >= m_layers[layer][neuronFrom]->getConections_count()) {
        std::cout << "getConWeight() invalid: " << layer << " " << neuronFrom << " "  <<  neuronTo << std::endl;
        return -1.0;
    }
    return m_layers[layer][neuronFrom]->m_outputWeights[neuronTo];
}

double Net::getNeuronValue(const unsigned int &layer, const unsigned int &neuron) const
{
    if(layer >= layerCount() || neuron >= neuronCountAt(layer) + 1 /*bias*/ ) {
        std::cout << "getNeuronValue() invalid: " << layer << " " << neuron << std::endl;
        return -1.0;
    }
    return m_layers[layer][neuron]->getOutputVal();
}


unsigned int Net::layerCount() const
{
    return getTopology().size();
}

unsigned int Net::neuronCountAt(const unsigned int &layer) const
{
    return getTopology().at(layer).neuronCount;
}

double Net::recentAverrageError() const
{
    return m_recentAverrageError;
}

void Net::feedForward(const double *input)
{
    for (unsigned i = 0; i < neuronCountAt(0); i++) {
        m_layers[0][i]->setOutputVal( input[i] );
    }

    for (unsigned layerNum = 1; layerNum < layerCount(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        Layer &thisLayer = m_layers[layerNum];

        bool is_softmax = thisLayer[0]->getAktiF() == LayerType::Aktivierungsfunktion::SMAX;
        double maxActivation = -std::numeric_limits<double>::infinity(); // Initialize maxActivation to negative infinity
        double logSumExp = 0.0; // Initialize logSumExp to zero

        for (unsigned n = 0; n < neuronCountAt(layerNum); ++n) {
            //Sum all connections....
            thisLayer[n]->aggregation(prevLayer, neuronCountAt(layerNum - 1) + 1);
        }

        if(is_softmax) {
            for (unsigned n = 0; n < neuronCountAt(layerNum); ++n)
                maxActivation = std::max(maxActivation, thisLayer[n]->getSumme_Aggregationsfunktion());
            for (unsigned n = 0; n < neuronCountAt(layerNum); ++n)
                logSumExp += std::exp(thisLayer[n]->getSumme_Aggregationsfunktion() - maxActivation);
            logSumExp = maxActivation + std::log(logSumExp);
        }

        // Now you can use sum_of_expo for softmax operation
        for (unsigned n = 0; n < neuronCountAt(layerNum); ++n) {
            thisLayer[n]->activation(logSumExp); // Subtract sum_of_expo
        }
    }
}

void Net::getResults(double *output) const
{
    unsigned outputlayer = layerCount() - 1;
    for (unsigned n = 0; n < neuronCountAt(outputlayer); ++n)  {
        output[n] = m_layers[outputlayer][n]->getOutputVal();
    }
}


void Net::backProp(double *targetVals, const double & eta, const double & alpha, const bool &batchLearning)
{
    Layer & outputLayer = m_layers[ this->layerCount() - 1]; // start from behind !
    unsigned output_neuron_count_ohne_bias =  neuronCountAt( this->layerCount() - 1 ) /*MACH ICHWEG!!!!! - 1*/ ;
    m_error = 0.0;

    for (unsigned n = 0; n < output_neuron_count_ohne_bias; ++n) {
        double delta = targetVals[n] - outputLayer[n]->getOutputVal();
        m_error += delta * delta;
    }
    m_error /= (double)output_neuron_count_ohne_bias;
    m_error = sqrt(m_error);
    m_recentAverrageError = (m_recentAverrageError * m_recentAverangeSmoothingFactor + m_error) / (m_recentAverangeSmoothingFactor + 1.0);


    // Calculate output layer gradiants
    for (unsigned n = 0; n < output_neuron_count_ohne_bias; ++n)
    {
        outputLayer[n]->calcOutputGradients(targetVals[n]);
    }

    //Calculate gradients on hidden layers: starts with max -2, because max -1 is last and wee dont need the output layer, so -2, and stop with 1: no inputl
    for (unsigned long Layernum = layerCount() - 2; Layernum > 0; --Layernum)
    {
        Layer & hidenLayer = m_layers[(Layernum)];
        Layer & nextLayer = m_layers[(Layernum + 1)];
        for (unsigned n = 0; n < neuronCountAt(Layernum) + 1; ++n) {
            hidenLayer[n]->calcHiddenGradients(nextLayer, neuronCountAt(Layernum + 1) + 1);
        }
    }

    //gewichte anpassen
    for (long Layernum = layerCount() -1 ; Layernum > 0; --Layernum)
    {
        Layer & layer = m_layers[Layernum];
        Layer & prevLayer = m_layers[Layernum -1];

        for (unsigned n = 0; n < neuronCountAt(Layernum) + 1; ++n)
        {
            layer[n]->updateInputWeights(prevLayer, neuronCountAt(Layernum -1) + 1, eta, alpha, batchLearning);
        }
    }
}


void Net::applyBatch()
{
    //gewichte anpassen
    for (long Layernum = layerCount() - 1 - 1/*skip last*/ ; Layernum >= /*update also first layer*/ 0; --Layernum)
    {
        Layer & layer = m_layers[Layernum];
        for (unsigned n = 0; n < neuronCountAt(Layernum) + 1; ++n)
        {
            layer[n]->applyBatch();
        }
    }
}
