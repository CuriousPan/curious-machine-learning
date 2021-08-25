import math
import copy

class HotNN():
    
    def __init__(self):
        self.data_x = None
        self.data_y = None

    def load_data(self, TRAIN_DATA_X, TRAIN_DATA_Y):
        self.data_x = TRAIN_DATA_X
        self.data_y = TRAIN_DATA_Y

    def evaluate(self, v, k):
        assert self.data_x is not None
        assert self.data_y is not None
        assert k >= 1 and k < len(self.data_x)

        distances = []
        for entry in self.data_x:
            #distances.append(self.__euclidianDistance__(v, entry))
            distances.append(self.__manhattanDistance__(v, entry))
        sortedDistances = sorted(distances)

        k_minimal_labels = self.__k_minimal_labels(distances, sortedDistances, self.data_y, k)
        return self.__most_frequent_element(k_minimal_labels)

    def test(self, x, y, k):
        success = 0
        for i in range(len(x)):
            result = self.evaluate(x[i], k)
            if (result == y[i]):
                success += 1
        #    print("Expected: " + str(y[i]) + ". Predicted: " + str(result))
        print("Accuracy: " + str(success/len(y)))
                
    def __most_frequent_element(self, _list):
        assert len(_list) > 0
        most_frequent_element = None
        most_frequent_counter = -1
        for i in range(len(_list)):
            element_counter = 0
            for k in range(len(_list)):
                if _list[i] == _list[k]:
                    element_counter += 1
            if element_counter > most_frequent_counter:
                most_frequent_element = _list[i]
                most_frequent_counter = element_counter
        if most_frequent_element == None:
            raise Exception
        return most_frequent_element

    def __k_minimal_labels(self, unsortedList, sortedList, _data_y, k):
        assert k > 0 and k <= len(sortedList)
        unsorted_list = copy.deepcopy(unsortedList)
        sorted_list = copy.deepcopy(sortedList)
        data_y = copy.deepcopy(_data_y)
        minimal_labels = []
        for i in range(k):
            current = sorted_list.pop(0)
            index_of_value = unsorted_list.index(current)
            minimal_labels.append(data_y.pop(index_of_value))
            unsorted_list.pop(index_of_value)
        return minimal_labels

    def __euclidianDistance__(self, v1, v2):
        assert len(v1) == len(v2)
        result = 0
        for i in range(len(v1)):
            result += math.pow((v1[i] - v2[i]), 2)
        return math.sqrt(result)

    def __manhattanDistance__(self, v1, v2):
        assert len(v1) == len(v2)
        result = 0
        for i in range(len(v1)):
            result += abs(v1[i] - v2[i])
        return result

#summer-717
