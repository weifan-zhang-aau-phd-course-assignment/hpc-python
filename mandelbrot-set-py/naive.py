import numpy as np

import common

version = "naive"

realsFile = "reals_{}.csv".format(version)
imagesFile = "images_{}.csv".format(version)
resultsFile = "results_{}.csv".format(version)


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


def calcResults():
    reals, images, reals2d, images2d = common.genRealsImages()

    results = np.zeros((len(images), len(reals)), dtype=np.float64)

    for i in range(len(reals2d)):
        for j in range(len(reals2d[0])):
            results[i][j] = M(reals2d[i][j], images2d[i][j])

    # for i in range(len(reals)):
    #     for j in range(len(images)):
    #         results[j][i] = M(reals[i], images[j])
    return reals, images, reals2d, images2d, results


def main():
    reals, images, _, _, results = calcResults()

    common.saveDataToFile(reals, images, results, realsFile, imagesFile,
                          resultsFile)
    reals, images, results = common.loadDataFromFile(realsFile, imagesFile,
                                                     resultsFile)
    common.plotHot(reals, images, results)


def onlyPlot():
    reals, images, results = common.loadDataFromFile(realsFile, imagesFile,
                                                     resultsFile)
    common.plotHot(reals, images, results)


if __name__ == "__main__":
    # main()
    onlyPlot()
