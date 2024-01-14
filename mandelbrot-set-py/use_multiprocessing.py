from concurrent import futures
import multiprocessing
import numpy as np

import common

version = "use_multiprocessing"

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


def splitListIdx(lst, m):
    n = len(lst)
    k = n // m  # size of every part
    remainder = n % m

    result = []
    start = 0
    for i in range(m):
        # put remainders in every part averagely
        size = k + 1 if i < remainder else k

        end = start + size
        result.append((start, end))
        start = end

    return result


def doM(reals2d, images2d, results, iStart: int, iEnd: int):
    for i in range(iStart, iEnd):
        for j in range(len(reals2d[0])):
            thisResult = M(reals2d[i][j], images2d[i][j])
            # For multiThreads, the memory is shared, so maybe we need a lock here, but this code can ensure that any 2 threads will not read or write a same position, so I think maybe we do not need a lodk here.
            results[i][j] = thisResult
    print("Finish [{}, {})]".format(iStart, iEnd))
    return results, iStart, iEnd


def calcResults(maxWorkers=num_cores, multiThreads=False, chunkPerWorker=3):

    # The minimum chunkPerWorker is 1.
    # If the chunkPerWorker is too large, the Python program will use too much memory and be OOM killed, so the user should be careful not to set the chunkPerWorker too big.
    if chunkPerWorker < 0:
        chunkPerWorker = 1

    reals, images, reals2d, images2d = common.genRealsImages()

    results = np.zeros((len(images), len(reals)), dtype=np.float64)

    # start to use multiple processors or threads to work
    print("maxWorkers:", maxWorkers)
    print("chunkPerWorker:", chunkPerWorker)
    pool = futures._base.Executor()
    if multiThreads:
        print("Using multiple threads.")
        pool = futures.ThreadPoolExecutor(max_workers=maxWorkers)
    else:
        print("Using multiple processors.")
        pool = futures.ProcessPoolExecutor(max_workers=maxWorkers)

    # split the tasks into {maxWorkers} parts
    splitIdxes = splitListIdx(reals2d, maxWorkers * chunkPerWorker)

    # do the {maxWorkers} parts of tasks in parallel
    print('Start to submit tasks')
    futureList = []
    for idxStartEnd in splitIdxes:
        print("submit task i in [{}, {})".format(idxStartEnd[0],
                                                 idxStartEnd[1]))
        futureList.append(
            pool.submit(doM, reals2d, images2d, results, idxStartEnd[0],
                        idxStartEnd[1]))

    print('Start to wait for tasks')
    pool.shutdown(wait=True)
    print('All tasks are done!')

    # in multiProcesses, the memory is not shared, so we need to put the result values in the results
    # in multiThreads, the memory is shared, so the result values have already been put in the results
    if not multiThreads:
        for future in futures.as_completed(futureList):
            thisResults, iStart, iEnd = future.result()
            for i in range(iStart, iEnd):
                results[i] = thisResults[i]

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
