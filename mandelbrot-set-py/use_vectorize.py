import numpy as np
from numba import vectorize

import common

version = "use_vectorize"

realsFile = "reals_{}.csv".format(version)
imagesFile = "images_{}.csv".format(version)
resultsFile = "results_{}.csv".format(version)


# @vectorize does not support 2 result values, so we need to have 2 functions mandelbrotRealFunc and mandelbrotImageFunc
@vectorize(['float64(float64, float64, float64, float64)'])
def mandelbrotRealFunc(inReal, inImage, cReal, cImage):
    outReal = inReal * inReal - inImage * inImage + cReal
    return outReal


@vectorize(['float64(float64, float64, float64, float64)'])
def mandelbrotImageFunc(inReal, inImage, cReal, cImage):
    outImage = cImage + 2 * inReal * inImage
    return outImage


# use vectorize to let Python do the same operation to all elements in the lists in parallel.
@vectorize(['float64(float64, float64)'])
def M(cReal, cImage):
    zReal = 0.0
    zImage = 0.0

    iotaC = common.iterationLimit

    # After this loop, if there is not a z larger than common.threshold, iotaC will be common.iterationLimit
    for i in range(1, common.iterationLimit + 1):
        oldZReal, oldZImage = zReal, zImage
        zReal = mandelbrotRealFunc(oldZReal, oldZImage, cReal, cImage)
        zImage = mandelbrotImageFunc(oldZReal, oldZImage, cReal, cImage)
        squareOfModulus = zReal * zReal + zImage * zImage
        if squareOfModulus > common.threshold * common.threshold:
            iotaC = i
            break

    return float(iotaC) / float(common.iterationLimit)


def calcResults():
    reals, images, reals2d, images2d = common.genRealsImages()

    results = np.zeros((len(images), len(reals)), dtype=np.float64)

    results = M(reals2d, images2d)

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
