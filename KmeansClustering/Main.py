from torchvision import datasets
import warnings
import kmeanFunction as func
import numpy as np
import pandas as pd



def main():
    print("\nProgram initialized")
    warnings.filterwarnings("ignore")
    train_data, test_data = func.downloadData(datasets.MNIST)
    train_reshaped, train_target_numpyArray = func.purifyData(train_data, test_data)
    accuracy_list, time_elapsed_list, inertia_list, homogeneity_list, total_clusters  = func.runProgram(train_reshaped, train_target_numpyArray)
    func.statsPrint(accuracy_list, time_elapsed_list, inertia_list, homogeneity_list, total_clusters)
    func.saveToCSV(accuracy_list, time_elapsed_list)
    print("End program\n")
main()

