#include "network.h"
#include "utils.h"
#include <iostream>
#include <random>
#include <iomanip>

int main()
{
    std::vector<std::vector<double>> dataset = Utils::readCSV("data/wheat-seeds-train.csv");
    std::vector<double> data_y;
    for (int i = 0; i < dataset.size(); ++i)
    {
        data_y.push_back(dataset[i].back() - 1); //because of dataset we have to -1 
        dataset[i].pop_back();
    }

    std::vector<std::vector<double>> minmax = Utils::minmax(dataset);
    std::vector<std::vector<double>> normalized = Utils::normalizeDataset(dataset, minmax);

    int numberOfInputs = dataset[0].size();
    int numberOfOuputs = Utils::numberUniqueElements(data_y);
    
    Network network(numberOfInputs);
    network.addHiddenLayer(5, &Utils::sigmoid);
    network.addHiddenLayer(3, &Utils::sigmoid);
    network.addOutputLayer(numberOfOuputs, &Utils::sigmoid, &Utils::squaredError);

    network.trainClassification(normalized, data_y, 2500, 0.15);

    double correct = 0;

    std::vector<std::vector<double>> test_dataset = Utils::readCSV("data/wheat-seeds-test.csv");
    std::vector<double> test_data_y;

    for (int i = 0; i < test_dataset.size(); ++i)
    {
        test_data_y.push_back(test_dataset[i].back() - 1); //because of dataset we have to -1 
        test_dataset[i].pop_back();
    }

    std::vector<std::vector<double>> minmax_y = Utils::minmax(dataset);
    std::vector<std::vector<double>> normalized_y = Utils::normalizeDataset(test_dataset, minmax_y);
    
    for (int i = 0; i < test_dataset.size(); ++i)
    {
        int prediction = network.predictClass(normalized_y[i]);
        std::cout << "Predicted: " << prediction << ". Real: " << test_data_y[i] << std::endl;

        if (prediction == test_data_y[i])
        {
            ++correct;
        }
    }
    std::cout << "Accuracy: " << correct/(double)(test_data_y.size()) << std::endl;
}
