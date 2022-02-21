#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <set>

namespace Utils
{
    double sigmoid(double x, bool derivative);
    double relu(double x, bool derivative);
    double tanh(double x, bool derivative);
    double dotProduct(std::vector<double> v1, std::vector<double> v2);
    //TODO(investigate mseLoss)
    double mseLoss(const std::vector<double>& predictions, const std::vector<double>& labels);
    double randomNumber(int min, int max);
    
    template<typename T>
    void printVector(const std::vector<T>& vector)
    {
        for (const T& element : vector)
        {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    //This is one is with 1/2
    double squaredError(double expected, double predicted, bool derivative);
    std::vector<std::vector<double>> minmax(std::vector<std::vector<double>> data);
    std::vector<std::vector<double>> normalizeDataset(std::vector<std::vector<double>> data, std::vector<std::vector<double>> minmax);

    std::vector<std::vector<double>> readCSV(std::string path);

    template<typename T>
    int numberUniqueElements(std::vector<T> v)
    {
        return std::set<T>(v.begin(), v.end()).size();
    }
}

#endif
