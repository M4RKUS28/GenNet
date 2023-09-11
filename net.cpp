#include "net.h"


Net::Net(const std::vector<LayerType> &topology)
    :  topology(topology)
{
    // create Net
    init();
}

Net::Net(const std::string &s_topology)
    :  topology(getTopologyFromStr(s_topology))
{
    // create Net
    init();
}


Net::Net(const Net *other)
    : Net(other->getTopology())
{
    // create Net
    init();
    // copy Net
    createCopyFrom(other);
}

Net::Net(const std::string filename, bool &ok)
{
    ok = load_from(filename);
}


void Net::init()
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
            m_layers[layerNum][neuronNum] = new Neuron(anzahl_an_neuronen_des_naechsten_Layers, neuronNum, topology.at(layerNum));
        }

        /*BIAS*/
        m_layers[layerNum][topology.at(layerNum).neuronCount]->setOutputVal(1);
    }
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


bool Net::load_from(const std::string &filename)
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
                init();

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

                if(part_s.length() != 3) {
                    std::cerr << "Invalid length Aggregationsfunktion: " << part_s << std::endl;
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
                    else {
                        std::cerr << "Invalid Aggregationsfunktion: " << part_s << std::endl;
                        break;
                    }
                }

                // Aktivierungsfunktion
            } else if(count == 2) {

                if(part_s.length() != 4) {
                    std::cerr << "Invalid Aktivierungsfunktion: " << part_s << std::endl;
                    break;
                }else {
                    if(part_s == "TANH")
                        lt.aktiF = LayerType::Aktivierungsfunktion::TANH;
                    else if(part_s == "RELU")
                        lt.aktiF = LayerType::Aktivierungsfunktion::RELU;
                    else if(part_s == "SMAX")
                        lt.aktiF = LayerType::Aktivierungsfunktion::SMAX;
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

std::string Net::getTopologyStr()
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

void Net::feedForward(const double *input)
{

    for (unsigned i = 0; i < neuronCountAt(0); i++) {
        m_layers[0][i]->setOutputVal( input[i] );
    }

    for (unsigned layerNum = 1; layerNum < layerCount() ; ++layerNum) {
        Layer &prevLayer = m_layers[layerNum -1];
        Layer &thisLayer = m_layers[layerNum   ];
        double sum_of_expo = 0.0;
        bool is_softmax = thisLayer[0]->getAktiF() == LayerType::Aktivierungsfunktion::SMAX;

        for (unsigned n = 0; n < neuronCountAt(layerNum) /*- 1*/ /* KEIN BIOS???!!! */; ++n) {
            thisLayer[n]->aggregation(prevLayer, neuronCountAt(layerNum -1) + 1 /*BIOS*/);
            if(is_softmax)
                sum_of_expo += std::exp( thisLayer[n]->getSumme_Aggregationsfunktion() );
        }
        for (unsigned n = 0; n < neuronCountAt(layerNum) /*- 1*/ /* KEIN BIOS???!!! */; ++n) {
            thisLayer[n]->activation(sum_of_expo);
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
