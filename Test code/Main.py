from torchvision import datasets
import warnings
import kmeanFunction as func
import time
import numpy as np

def runClustering(train_data, test_data, train_reshaped, train_target_numpyArray, test_reshaped, test_target_numpyArray, train_numpyArray, test_numpyArray):
    
    start_time = time.time()
    kmeansLabels = func.generateClusters(train_reshaped, test_target_numpyArray)
    reference_labels = func.retrieveInfo(kmeansLabels,train_target_numpyArray)
    number_labels = func.assignPredictions(kmeansLabels, reference_labels)
    accuracy = func.printPerformanceMetrics(reference_labels, number_labels, train_target_numpyArray)
    #func.printSpecificPicture(train_numpyArray, 0)
    time_elapsed = time.time()-start_time
    return time_elapsed, accuracy

def main():
    print("RUN\n\n")
    warnings.filterwarnings("ignore")
    train_data, test_data = func.downloadData(datasets.MNIST)
    train_reshaped, train_target_numpyArray, test_reshaped, test_target_numpyArray, train_numpyArray, test_numpyArray = func.purifyData(train_data, test_data)
    
    iterations = 10
    i=iterations
    time_elapsed_list = []
    accuracy_list = []

    sum_time = 0

    while i>0:
        time_elapsed, accuracy = runClustering(train_data, test_data, train_reshaped, train_target_numpyArray, test_reshaped, test_target_numpyArray, train_numpyArray, test_numpyArray)
        print("STATS: ", i, "Time elapsed: ", time_elapsed, "Accuracy: ", accuracy)
        time_elapsed_list.append(time_elapsed)
        accuracy_list.append(accuracy)
        i-=1
    
    print("\n\nTIME ELAPSED: ")
    print(time_elapsed_list, "\n")
    print("ACCURACY: ")
    print(accuracy_list, "\n")
    print(type(accuracy_list))
    print("AVERAGE ACCURACY:")
    print(round(sum(accuracy_list)/len(accuracy_list),2))
    print("AVERAGE TIME:")
    print(round(sum(time_elapsed_list)/len(time_elapsed_list),2))

    

    print("\nFINISHED")

main()

