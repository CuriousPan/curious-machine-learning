#ifndef LAYER_CPP
#define LAYER_CPP

#include "layer.h"

Layer::Layer(unsigned int n_neurons, unsigned int m_inputs, double (*activationFunction)(double, bool), bool isOutputLayer, double (*lossFunction)(double, double, bool)) : m_outputLayer(isOutputLayer)
{
    for (unsigned int i = 0; i < n_neurons; ++i)
    {
        m_neurons.push_back(Neuron(m_inputs, activationFunction, lossFunction));
    }
}

Layer::Layer(int i)
{
    if (i == 1)
    {
        m_neurons.push_back(Neuron(1));
    }
    else if (i == 2)
    {
        m_neurons.push_back(Neuron(2));
        m_neurons.push_back(Neuron(3));
        m_outputLayer = true;
    }
}

const std::vector<Neuron>& Layer::neurons() const
{
    return m_neurons;
}

std::vector<Neuron>& Layer::neurons()
{
    return m_neurons;
}

bool Layer::isOutputLayer() const
{
    return m_outputLayer;
}

std::vector<double> Layer::activateLayer(const std::vector<double>& input)
{
    std::vector<double> output;
    for (auto& neuron : m_neurons)
    {
        output.push_back(neuron.activate(input));
    }
    return output;
}

const Neuron& Layer::operator[](unsigned int index) const
{
    return m_neurons.at(index);
}

Neuron& Layer::operator[](unsigned int index)
{
    return m_neurons.at(index);
}

std::ostream &operator<<(std::ostream &ostream, const Layer &layer)
{
    for (const auto &neuron : layer.m_neurons)
    {
        ostream << neuron << std::endl;
    }
    ostream << "\n" << std::endl;
    return ostream;
}

#endif 
