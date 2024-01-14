from concurrent import futures
import multiprocessing
import numpy as np

import common

version = "use_multiprocessing_bad_performance"

realsFile = "reals_{}.csv".format(version)
imagesFile = "images_{}.csv".format(version)
resultsFile = "results_{}.csv".format(version)

num_cores = multiprocessing.cpu_count()


def mandelbrotFunc(inReal: float, inImage: float, cReal: float,
                   cImage: float) -> tuple[float, float]:
    outReal = inReal * inReal - inImage * inImage + cReal
    outImage = cImage + 2 * inReal * inImage
    return outReal, outImage


def M(cReal: float, cImage: float) -> float:
    zReal = 0.0
    zImage = 0.0

    iotaC = common.iterationLimit

    # After this loop, if there is not a z larger than common.threshold, iotaC will be common.iterationLimit
    for i in range(1, common.iterationLimit + 1):
        zReal, zImage = mandelbrotFunc(zReal, zImage, cReal, cImage)
        squareOfModulus = zReal * zReal + zImage * zImage
        if squareOfModulus > common.threshold * common.threshold:
            iotaC = i
            break

    return float(iotaC) / float(common.iterationLimit)


def doMWithIdx(i: int, j: int, cReal: float, cImage: float):
    return i, j, M(cReal, cImage)


def calcResults(maxWorkers=num_cores, multiThreads=False):

    reals, images, reals2d, images2d = common.genRealsImages()

    results = np.zeros((len(images), len(reals)), dtype=np.float64)

    # start to use multiple processors or threads to work
    print("maxWorkers:", maxWorkers)
    pool = futures._base.Executor()
    if multiThreads:
        print("Using multiple threads.")
        pool = futures.ThreadPoolExecutor(max_workers=maxWorkers)
    else:
        print("Using multiple processors.")
        pool = futures.ProcessPoolExecutor(max_workers=maxWorkers)
    futureList = []

    print('Start to submit tasks')
    for i in range(len(reals2d)):
        for j in range(len(reals2d[0])):
            futureList.append(
                pool.submit(doMWithIdx, i, j, reals2d[i][j], images2d[i][j]))

    print('Start to wait for tasks')
    pool.shutdown(wait=True)
    print('All tasks are done!')

    for future in futures.as_completed(futureList):
        i, j, result = future.result()
        results[i][j] = result

    print('Finish getting results')

    return reals, images, reals2d, images2d, results


def main():
    reals, images, _, _, results = calcResults()

    common.saveDataToFile(reals, images, results, realsFile, imagesFile,
                          resultsFile)
    reals, images, results = common.loadDataFromFile(realsFile, imagesFile,
                                                     resultsFile)
    common.plotHot(reals, images, results)


if __name__ == "__main__":
    main()
