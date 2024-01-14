import matplotlib.pyplot as plt
import numpy as np

nameMethodsAll = [
    'naive', 'vectorize', 'guvectorize', 'multi-thread 1 worker',
    'multi-thread 2 worker', 'multi-thread 3 worker', 'multi-thread 4 worker',
    'multi-thread 5 worker', 'multi-thread 6 worker', 'multi-thread 7 worker',
    'multi-thread 8 worker', 'multi-process 1 worker',
    'multi-process 2 worker', 'multi-process 3 worker',
    'multi-process 4 worker', 'multi-process 5 worker',
    'multi-process 6 worker', 'multi-process 7 worker',
    'multi-process 8 worker'
]

testMethodsAll = [
    'naive.calcResults()', 'use_vectorize.calcResults()',
    'use_guvectorize.calcResults()',
    'use_multiprocessing.calcResults(1, True)',
    'use_multiprocessing.calcResults(2, True)',
    'use_multiprocessing.calcResults(3, True)',
    'use_multiprocessing.calcResults(4, True)',
    'use_multiprocessing.calcResults(5, True)',
    'use_multiprocessing.calcResults(6, True)',
    'use_multiprocessing.calcResults(7, True)',
    'use_multiprocessing.calcResults(8, True)',
    'use_multiprocessing.calcResults(1, False)',
    'use_multiprocessing.calcResults(2, False)',
    'use_multiprocessing.calcResults(3, False)',
    'use_multiprocessing.calcResults(4, False)',
    'use_multiprocessing.calcResults(5, False)',
    'use_multiprocessing.calcResults(6, False)',
    'use_multiprocessing.calcResults(7, False)',
    'use_multiprocessing.calcResults(8, False)'
]

testMethodsPart = [
    'naive.calcResults()',
    'use_vectorize.calcResults()',
    'use_guvectorize.calcResults()',
    # 'use_multiprocessing_bad_performance.calcResults()',
    'use_multiprocessing.calcResults()',
    'use_multiprocessing.calcResults(multiThreads=True)'
]

###
# These parameters are according to the project description.

realMax = 1.0
realMin = -2.0
imageMax = 1.5
imageMin = -1.5

# This is like the 'resolution', and can affect the execution time of the project. The project description asks us to select the 2 parameters according to the computational resources I have available.
numReal = 1000
numImage = 1000

threshold = 2
iterationLimit = 100

###


# generate the list of reals and images
def genRealsImages():
    stepReal = (realMax - realMin) / numReal
    reals = []
    i = realMin
    while i <= realMax:
        reals.append(i)
        i += stepReal

    stepImage = (imageMax - imageMin) / numImage
    images = []
    i = imageMin
    while i <= imageMax:
        images.append(i)
        i += stepImage

    outReals = np.array(reals, dtype=np.float64)
    outImages = np.array(images, dtype=np.float64)

    reals2d = np.zeros((len(outImages), len(outReals)), dtype=np.float64)
    images2d = np.zeros((len(outImages), len(outReals)), dtype=np.float64)
    for i in range(len(outReals)):
        for j in range(len(outImages)):
            reals2d[j][i] = outReals[i]
            images2d[j][i] = outImages[j]

    return outReals, outImages, reals2d, images2d


def saveDataToFile(realsData, imagesData, resultsData, realsFileName,
                   imagesFileName, resultsFileName):
    np.savetxt(realsFileName, realsData, delimiter=',', fmt='%f')
    np.savetxt(imagesFileName, imagesData, delimiter=',', fmt='%f')
    np.savetxt(resultsFileName, resultsData, delimiter=',', fmt='%f')


def loadDataFromFile(realsFileName, imagesFileName, resultsFileName):
    reals = np.loadtxt(realsFileName, delimiter=',', dtype=np.float64)
    images = np.loadtxt(imagesFileName, delimiter=',', dtype=np.float64)
    results = np.loadtxt(resultsFileName, delimiter=',', dtype=np.float64)
    return reals, images, results


def plotHot(reals, images, results):
    plt.imshow(results,
               cmap='hot',
               origin='lower',
               extent=[realMin, realMax, imageMin, imageMax])
    plt.colorbar()

    x_tick_interval = int(len(reals) / 10)
    y_tick_interval = int(len(images) / 10)
    plt.xticks(reals[::x_tick_interval])
    plt.yticks(images[::y_tick_interval])

    plt.title("Mandelbrot Set Plotting")

    plt.show()
