from torchvision import datasets
import warnings
import kmeanFunction as func


def main():
    print("RUN")
    warnings.filterwarnings("ignore")
    train_data, test_data = func.downloadData(datasets.MNIST)
    train_reshaped, train_target_numpyArray, test_reshaped, test_target_numpyArray, train_numpyArray, test_numpyArray = func.purifyData(train_data, test_data)
    kmeansLabels = func.generateClusters(train_reshaped, test_target_numpyArray)
    reference_labels = func.retrieveInfo(kmeansLabels,train_target_numpyArray)
    number_labels = func.assignPredictions(kmeansLabels, reference_labels)
    func.printPerformanceMetrics(reference_labels, number_labels, train_target_numpyArray)
    func.printSpecificPicture(train_numpyArray, 0)

    print("FINISHED")

if __name__ == '__main__':
    main()
