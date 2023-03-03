from torchvision import datasets
import warnings
import kmeanFunction as func
import numpy as np



def main():
    print("\nProgram initialized")
    warnings.filterwarnings("ignore")
    train_data, test_data = func.downloadData(datasets.MNIST)
    train_reshaped, train_target_numpyArray, test_reshaped, test_target_numpyArray, train_numpyArray, test_numpyArray = func.purifyData(train_data, test_data)
    accuracy_list, time_elapsed_list = func.runProgram(train_data, test_data, train_reshaped, train_target_numpyArray, test_reshaped, test_target_numpyArray, train_numpyArray, test_numpyArray)
    func.statsPrint(accuracy_list, time_elapsed_list)
    print("End program\n")
main()

