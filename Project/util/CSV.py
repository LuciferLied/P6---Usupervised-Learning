import os
import csv


def saveToCSV(values):
    
    rootdir = os.getcwd()

    for subdir, dirs, files in os.walk(rootdir):
        
        filepath = subdir + os.sep

        if filepath.__contains__("dataanalysis"):
            filepath = filepath + "CSVstats.csv"
            with open(filepath, 'a', newline='') as CSVstats:
                writer = csv.writer(CSVstats)
                writer.writerow(values)

def delCSVContent():
    rootdir = os.getcwd()

    for subdir, dirs, files in os.walk(rootdir):
        
        filepath = subdir + os.sep

        if filepath.__contains__("dataanalysis"):
            filepath = filepath + "CSVstats.csv"
            with open(filepath, 'w', newline='') as CSVstats:
                CSVstats.write('')

def lastLineCSV():
    rootdir = os.getcwd()

    for subdir, dirs, files in os.walk(rootdir):
        
        filepath = subdir + os.sep

        if filepath.__contains__("dataanalysis"):
            filepath = filepath + "CSVstats.csv"
            with open(filepath, "r") as CSVstats:
                lines = CSVstats.readlines()
                last_line = lines[-1]
                return last_line