#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <iostream>
#include "utils.h"

class Neuron
{
public:
    Neuron(unsigned int m_inputs, double (*activationFunction)(double, bool), double (*lossFunction)(double, double, bool) = nullptr);
    
    Neuron(int i);

    std::vector<double> &weights();
    double bias() const;
    double lastOutput() const;
    double delta() const;
    double lastNet() const;

    void setDelta(double value);
    void setBias(double value);

    double activate(const std::vector<double>& input);

    double activationFunction(double value) const;
    double derivativeActivationFunction(double value) const;

    double lossFunction(double expected, double predicted) const;
    double derivativeLossFunction(double expected, double predicted) const;

    friend std::ostream &operator<<(std::ostream &ostream, const Neuron &neuron);
    const double &operator[](unsigned int index) const;
    double &operator[](unsigned int index);


private:
    std::vector<double> m_weights;
    double m_bias;
    double m_lastOutput;
    double m_lastNet;
    double m_delta;
    double (*m_activationFunction)(double, bool);
    double (*m_lossFunction)(double, double, bool);
};

#endif
