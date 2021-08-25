import Model

IRIS_TRAIN_PATH = "datasets/iris_train.csv"
IRIS_TEST_PATH = "datasets/iris_test.csv"
HEART_TRAIN_PATH = "datasets/heart_train.csv"
HEART_TEST_PATH = "datasets/heart_test.csv"

def main():
    train_data, train_y = load_data(IRIS_TRAIN_PATH)
    test_data, test_y = load_data(IRIS_TEST_PATH)

    model = Model.HotNN()
    model.load_data(train_data, train_y)
    for i in range(1, 100):
        print("k:", i, end = " ") 
        model.test(test_data, test_y, i)

def load_data(path):
    file = open(path)
    lines = file.readlines()
    file.close()

    #I had some issues with reading files of specific endcoding.
    #that's a wacky solution, but none of suggested worked
    for i in range(len(lines)):
        lines[i] = lines[i].split(",")
        lines[i][-1] = lines[i][-1].replace('\n', '')

    features_name = lines.pop(0)
    features_name.pop(-1)
   
    labels = []
    for i in range(len(lines)):
        labels.append(lines[i].pop())
        for k in range(len(lines[i])):
            lines[i][k] = float(lines[i][k])
    
    return lines, labels

if __name__ == "__main__":
    main()

#summer-717
