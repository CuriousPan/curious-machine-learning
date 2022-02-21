#ifndef NEURON_CPP
#define NEURON_CPP

#include "neuron.h"
#include "utils.h"
#include <assert.h>

#define ACTIVATION_FUNCTION(x) m_activationFunction(x, false)
#define DERIVATIVE_ACTIVATION_FUNCTION(x) m_activationFunction(x, true)
#define LOSS_FUNCTION(x, y) m_lossFunction(x, y, false)
#define DERIVATIVE_LOSS_FUNCTION(x, y) m_lossFunction(x, y, true)

Neuron::Neuron(unsigned int m_inputs, double (*activationFunction)(double, bool), double (*lossFunction)(double, double, bool)) : m_activationFunction(activationFunction), m_lossFunction(lossFunction)
{
    for (int i = 0; i < m_inputs; ++i)
    {
        m_weights.push_back(Utils::randomNumber(0, 1));
    }
    m_bias = Utils::randomNumber(0, 1);
}

Neuron::Neuron(int i)
{
    if (i == 1) {
        m_lastOutput = 0.7105668883115941;
        m_weights.push_back(0.13436424411240122);
        m_weights.push_back(0.8474337369372327);
        m_bias = 0.763774618976614;
    } else if (i == 2) {
        m_lastOutput = 0.6213859615555266;
        m_weights.push_back(0.2550690257394217);
        m_bias = 0.49543508709194095;
    } else if (i == 3) {
        m_lastOutput = 0.6573693455986976;
        m_weights.push_back(0.4494910647887381);
        m_bias = 0.651592972722763;
    }
    m_activationFunction = &Utils::sigmoid;
    m_lossFunction = &Utils::squaredError;
}

std::vector<double>& Neuron::weights()
{
    return m_weights;
}

double Neuron::lastOutput() const
{
    return m_lastOutput;
}

double Neuron::delta() const
{
    return m_delta;
}

void Neuron::setDelta(double value)
{
    m_delta = value;
}

double Neuron::bias() const
{
    return m_bias;
}

void Neuron::setBias(double value)
{
    m_bias = value;
}

double Neuron::lastNet() const
{
    return m_lastNet;
}

double Neuron::activate(const std::vector<double>& input)
{
    assert(input.size() == m_weights.size());
    double sum = m_bias;
    for (unsigned int i = 0; i < input.size(); ++i)
    {
        sum += input[i] * m_weights[i];
    }
    m_lastNet = sum;
    m_lastOutput = ACTIVATION_FUNCTION(sum);
    return m_lastOutput;
}

double Neuron::activationFunction(double value) const
{
    return ACTIVATION_FUNCTION(value);
}

double Neuron::derivativeActivationFunction(double value) const
{
    return DERIVATIVE_ACTIVATION_FUNCTION(value);
}

double Neuron::lossFunction(double expected, double predicted) const
{
    return LOSS_FUNCTION(expected, predicted);
}

double Neuron::derivativeLossFunction(double expected, double predicted) const
{
    return DERIVATIVE_LOSS_FUNCTION(expected, predicted);
}

std::ostream& operator<<(std::ostream &ostream, const Neuron &neuron)
{
    ostream << "Weights: [";
    if (neuron.m_weights.size() > 1)
    {
        for (unsigned int i = 0; i < neuron.m_weights.size() - 1; ++i)
        {
            ostream << neuron.m_weights[i] << ", ";
        }
        ostream << neuron.m_weights[neuron.m_weights.size() - 1] << "]. ";
    }
    else
    {
        ostream << neuron.m_weights[0] << "] " << std::endl;
    }
    ostream << "Bias: " << neuron.m_bias << ". ";

    //TODO(Investigate if it's possible to use function pointers in switch case)
    ostream << "Activation function: ";
    if (neuron.m_activationFunction == &Utils::sigmoid)
    {
        ostream << "Sigmoid." << std::endl;
    }
    else if (neuron.m_activationFunction == &Utils::relu)
    {
        ostream << "ReLu." << std::endl;
    }
    else if (neuron.m_activationFunction == &Utils::tanh)
    {
        ostream << "tanh." << std::endl;
    }
    else
    {
        ostream << "Some other activation function." << std::endl;
    }
    ostream << "Delta: " << neuron.m_delta << std::endl;
    ostream << "Last output: " << neuron.m_lastOutput << std::endl;
    return ostream;
}
const double& Neuron::operator[](unsigned int index) const
{
    return m_weights.at(index);
}

double& Neuron::operator[](unsigned int index)
{
    return m_weights.at(index);
}

#endif

