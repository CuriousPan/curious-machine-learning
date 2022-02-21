#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"
#include "utils.h"
#include <assert.h>
#include <iostream>
#include <math.h>
#include <algorithm>

Network::Network(unsigned int n_inputs) : m_n_inputs(n_inputs)
{
    
}

void Network::makeFake()
{
    m_layers.push_back(1);
    m_layers.push_back(2);
    m_n_outputs = 2;
}

void Network::addHiddenLayer(unsigned int n_neurons, double (*activationFunction)(double, bool))
{
    if (m_n_outputs == -1) //no output layer added
    {
        unsigned int numberOfInputsToEachNeuron = m_layers.size() > 0 ? m_layers.back().neurons().size() : m_n_inputs;
        m_layers.push_back(Layer(n_neurons, numberOfInputsToEachNeuron, activationFunction));
    }
    else
    {
        throw std::logic_error("Cannot add hidden layer after adding output layer.");
    }
}


void Network::addOutputLayer(unsigned int n_neurons, double (*activationFunction)(double, bool), double (*lossFunction)(double, double, bool))
{
    assert(m_layers.size() > 0); // checks if there are any hidden layers, as otherwise we cannot add an output layer (in my system).
    m_layers.push_back(Layer(n_neurons, m_layers.back().neurons().size(), activationFunction, true, lossFunction));
    m_n_outputs = n_neurons;
}

std::vector<double> Network::forwardPropagate(const std::vector<double> &input)
{
    assert(input.size() == m_n_inputs);
    std::vector<double> newInputs = input;
    for (auto& layer : m_layers)
    {
        newInputs = layer.activateLayer(newInputs);
    }
    return newInputs;
}

void Network::backPropagate(const std::vector<double>& expectedResult)
{
    for (int l = m_layers.size() - 1; l > -1; --l)
    {
        std::vector<double> dEtotal_dOut; // From my point of view, the article uses pretty poor naming. I guess mine is more clear. At least for me.
        Layer& layer = m_layers.at(l);
        if (layer.isOutputLayer())
        {
            for (int n = 0; n < layer.neurons().size(); ++n)
            {
                Neuron &neuron = layer[n];
                dEtotal_dOut.push_back(neuron.derivativeLossFunction(expectedResult.at(n), neuron.lastOutput()));
            }
        }
        else // hidden layer
        {
            for (int n = 0; n < layer.neurons().size(); ++n)
            {
                double error = 0.0;
                for (int n_ = 0; n_ < m_layers.at(l + 1).neurons().size(); ++n_) {
                    Neuron &neuron = m_layers.at(l + 1)[n_];
                    error += neuron.delta() * neuron[n];
                }
                dEtotal_dOut.push_back(error);
            }
        }
        for (int n = 0; n < layer.neurons().size(); ++n)
        {
            Neuron &neuron = layer[n];
            neuron.setDelta(dEtotal_dOut.at(n) * neuron.derivativeActivationFunction(neuron.lastNet()));
        }
    }
}

unsigned int Network::numberOfInputs() const
{
    return m_n_inputs;
}

unsigned int Network::numberOfOutputs() const
{
    return m_n_outputs;
}

const std::vector<Layer>& Network::layers() const
{
    return m_layers;
}

void Network::trainClassification(const std::vector<std::vector<double>>& train_X, const std::vector<double>& train_Y, unsigned int epochs, double learningRate)
{
    assert(train_X.size() == train_Y.size());
    for (unsigned int e = 0; e < epochs; ++e)
    {
        double epochError = 0.0;
        for (unsigned int i = 0; i < train_X.size(); ++i)
        {
            std::vector<double> outputs = forwardPropagate(train_X[i]);

            std::vector<double> expected;
            expected.resize(m_n_outputs, 0.0);
            expected[train_Y[i]] = 1;

            //TODO(consider this part of code)
            for (int b = 0; b < m_n_outputs; ++b)
            {
                epochError += m_layers.back()[0].lossFunction(expected[b], outputs[b]);
            }
            backPropagate(expected);
            updateWeights(learningRate, train_X[i]);
        }
        std::cout << "Epoch: " << e << ". Loss: " << epochError << ".\n";
    }
}

void Network::updateWeights(double learningRate, const std::vector<double>& networkInputs)
{
    for (unsigned int l = 0; l < m_layers.size(); ++l)
    {
        std::vector<double> previousLayerOutputs;
        if (l == 0) // If it's an input layer.
        {
            previousLayerOutputs = networkInputs;
        }
        else // Not an input layer.
        {
            for (int n = 0; n < m_layers[l - 1].neurons().size(); ++n)
            {
                previousLayerOutputs.push_back(m_layers[l - 1][n].lastOutput());
            }
        }

        for (int n = 0; n < m_layers[l].neurons().size(); ++n)
        {
            Neuron &neuron = m_layers[l][n];
            for (int i = 0; i < neuron.weights().size(); ++i)
            {
                neuron[i] -= neuron.delta() * learningRate * previousLayerOutputs[i];
            }
            neuron.setBias(neuron.bias() - neuron.delta() * learningRate);
        }
    }
}

unsigned int Network::predictClass(std::vector<double> input)
{
    assert(m_layers.size() > 0 && m_layers.back().isOutputLayer() && input.size() == m_n_inputs); //check if network topology is ready to make predictions
    std::vector<double> result = forwardPropagate(input);
    std::vector<double>::iterator iterator = std::max_element(result.begin(), result.end());
    return std::distance(result.begin(), iterator);
}

std::ostream &operator<<(std::ostream &ostream, const Network &network)
{
    std::cout << "Input layer: " << network.m_n_inputs << "\n" << std::endl;
    int i = 0;
    for (const auto &layer : network.m_layers)
    {
        if (layer.isOutputLayer())
        {
            ostream << "Output layer: \n" << layer << std::endl;
        }
        else
        {
            ostream << "Hidden layer: " << ++i << "\n" << layer << std::endl;
        }
    }
    ostream << "\n" << std::endl;
    return ostream;
}

#endif
