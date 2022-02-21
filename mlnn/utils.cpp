#ifndef UTILS_CPP
#define UTILS_CPP

#include "utils.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <time.h>
#include <typeinfo>
#include <fstream>
#include <sstream>

double Utils::sigmoid(double x, bool derivative)
{
    if (derivative) {
        return (1.0/(1.0 + exp(-x))) * (1.0 - 1.0/(1.0 + exp(-x)));
    }
    return 1.0/(1.0 + exp(-x));
}

double Utils::relu(double x, bool derivative)
{
    if (derivative) {
        if (x <= 0.0) {
            return 0.0;
        } else {
            return 1.0;
        }
    }
    return std::max(0.0, x);
}

double Utils::tanh(double x, bool derivative)
{
    if (derivative) {
        return 1.0 - std::pow((exp(x) - exp(-x))/(exp(x) + exp(-x)), 2.0);
    }
    return (exp(x) - exp(-x))/(exp(x) + exp(-x));
}

//dot product works with doubles only, as well as the whole system
double Utils::dotProduct(std::vector<double> v1, std::vector<double> v2)
{
    assert(v1.size() == v2.size());
    double result = 0.0;
    for (std::size_t i = 0; i < v1.size(); ++i)
    {
        result += v1.at(i) * v2.at(i);
    }
    return result;
}

double Utils::mseLoss(const std::vector<double>& predictions, const std::vector<double>& labels)
{
    assert(predictions.size() == labels.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i)
    {
        sum += std::pow(labels[i] - predictions[i], 2);
    }
    return sum/predictions.size();
}

double Utils::randomNumber(int min, int max)
{
    //TODO(investigate if it's a good approach
    static std::mt19937 rng{static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())};
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double Utils::squaredError(double expected, double predicted, bool derivative = false)
{
    if (derivative)
    {
        return -(expected - predicted);
    }
    return std::pow(expected - predicted, 2)/2;
}

std::vector<std::vector<double>> Utils::minmax(std::vector<std::vector<double>> data)
{
    assert(data.size() != 0);
    std::vector<std::vector<double>> result;
    for (int column = 0; column < data[0].size(); ++column)
    {
        double maxForColumn = data[0][column];
        double minForColumn = data[0][column];
        for (int row = 0; row < data.size(); ++row)
        {
            if (data[row][column] > maxForColumn)
            {
                maxForColumn = data[row][column];
            }

            if (data[row][column] < minForColumn)
            {
                minForColumn = data[row][column];
            }
        }
        result.push_back({minForColumn, maxForColumn});
    }
    return result;
}

std::vector<std::vector<double>> Utils::normalizeDataset(std::vector<std::vector<double>> data, std::vector<std::vector<double>> minmax)
{
    std::vector<std::vector<double>> result;
    for (int row = 0; row < data.size(); ++row)
    {
        result.push_back({});
        for (int i = 0; i < data[row].size(); ++i)
        {
            double a = (minmax[i][1] - minmax[i][0]) == 0.0 ? 1 : (minmax[i][1] - minmax[i][0]);
            result[result.size() - 1].push_back((data[row][i] - minmax[i][0]) / a);
        }
    }
    return result;
}

std::vector<std::vector<double>> Utils::readCSV(std::string path)
{
    std::ifstream file(path);
    if (file.is_open())
    {
        std::vector<std::vector<double>> result;
        std::string line;
        int i = 0;
        while (std::getline(file, line))
        {
            result.push_back({});
            std::stringstream stream(line);
            std::string number;
            char separator = ',';
            while (std::getline(stream, number, separator))
            {
                result.back().push_back(std::stod(number));
            }
        }
        file.close();
        return result;
    }
    std::cout << "Couldn't open the file." << std::endl;
    return {};
}


#endif
