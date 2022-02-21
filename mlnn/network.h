#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

class Network
{

public:
    Network(unsigned int n_inputs);
    void makeFake();

    void addHiddenLayer(unsigned int n_neurons, double (*activationFunction)(double, bool));
    void addOutputLayer(unsigned int n_neurons, double (*activationFunction)(double, bool), double (*lossFunction)(double, double, bool));

    unsigned int numberOfInputs() const;
    unsigned int numberOfOutputs() const;
    const std::vector<Layer>& layers() const;

    void trainClassification(const std::vector<std::vector<double>>& train_X, const std::vector<double>& train_Y, unsigned int epochs, double learningRate);

    unsigned int predictClass(std::vector<double> input);

    friend std::ostream &operator<<(std::ostream &ostream, const Network &network);

private:



private:
    std::vector<Layer> m_layers;
    unsigned int m_n_inputs = -1;
    unsigned int m_n_outputs = -1;

    void backPropagate(const std::vector<double>& expectedResult);
    void updateWeights(double learningRate, const std::vector<double>& inputs);
    std::vector<double> forwardPropagate(const std::vector<double> &networkInputs);
};

#endif
