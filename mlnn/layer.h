#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer
{
public:
    Layer(unsigned int n_neurons, unsigned int m_inputs, double (*activationFunction)(double, bool), bool isHiddenLayer = false, double (*lossFunction)(double, double, bool) = nullptr);

    Layer(int i);

    const std::vector<Neuron>& neurons() const;
    std::vector<Neuron>& neurons();

    bool isOutputLayer() const;

    std::vector<double> activateLayer(const std::vector<double>& input);

    friend std::ostream &operator<<(std::ostream &ostream, const Layer &layer);
    const Neuron& operator[](unsigned int index) const;
    Neuron& operator[](unsigned int index);

private:
    std::vector<Neuron> m_neurons;
    bool m_outputLayer = false;
};

#endif 
