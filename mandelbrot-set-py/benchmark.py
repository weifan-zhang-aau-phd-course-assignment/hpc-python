import datetime
import csv
import matplotlib.pyplot as plt

import common
import naive
import use_vectorize
import use_guvectorize
import use_multiprocessing
import use_multiprocessing_bad_performance

methods = common.testMethodsAll
methodsNames = common.nameMethodsAll

csvFileName = 'benchmark.csv'


# write the methods names and duration into a csv file
def writeCsv(names, durations):
    with open(csvFileName, 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=",")
        for i in range(len(names)):
            writer.writerow([names[i], durations[i]])


# read data from csv file and plot the data
def plotResults():
    # read data from csv file
    names = []
    durations = []
    with open(csvFileName, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=",")
        for row in reader:
            names.append(row[0])
            durations.append(float(row[1]))

    # plot the data
    fig, ax = plt.subplots()
    ax.bar(names, durations)
    ax.set_xlabel('Methods')
    ax.set_ylabel('Time used (second)')
    plt.title("Benchmark")

    plt.xticks(rotation=45)
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, ha='right')

    ax.grid(axis='y', linestyle='--')

    # ensure that the x label text can be shown in the picture
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    # initialize the list with the length of len(methods)
    durations = []
    for i in range(len(methods)):
        durations.append(datetime.datetime.now() - datetime.datetime.now())

    # benchmark
    for i in range(len(methods)):
        _, _, _, _, _ = eval(methods[i])
        timeBefore = datetime.datetime.now()
        _, _, _, _, _ = eval(methods[i])
        timeAfter = datetime.datetime.now()

        durations[i] = (timeAfter - timeBefore).total_seconds()
        print('"{}" uses time: {} s'.format(methods[i], durations[i]))

    # print results
    print()
    print("results:")
    for i in range(len(methods)):
        print('Method: {:<45}\t Name: {:<25}\t Time used: {} s'.format(
            methods[i], methodsNames[i], durations[i]))

    # write the results in csv file
    writeCsv(methodsNames, durations)
