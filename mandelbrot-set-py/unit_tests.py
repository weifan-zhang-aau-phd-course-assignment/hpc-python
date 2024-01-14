import unittest
import numpy as np

import common
import naive
import use_vectorize
import use_guvectorize
import use_multiprocessing
import use_multiprocessing_bad_performance


def wrongCalcResults():
    reals, images, reals2d, images2d, results = naive.calcResults()

    reals = np.random.random(len(reals))
    images = np.random.random(len(images))
    reals2d = np.random.random(reals2d.shape)
    images2d = np.random.random(images2d.shape)
    results = np.random.random(results.shape)

    return reals, images, reals2d, images2d, results


class testAll(unittest.TestCase):

    def setUp(self) -> None:
        self.calcResultsMethods = common.testMethodsPart

    # ensure that the `naive` method of Mandelbrot Set is correct.
    def testNaiveM(self):
        cReals = np.random.uniform(common.realMin, common.realMax, 5)
        cImages = np.random.uniform(common.imageMin, common.imageMax, 5)

        for cReal in cReals:
            for cImage in cImages:
                result = naive.M(cReal, cImage)
                iotaC = int(result * float(common.iterationLimit))
                # print("iotaC float is:", result * float(common.iterationLimit))
                # print("iotaC int is:", iotaC)

                zReal = 0.0
                zImage = 0.0
                i = 1
                while i < iotaC:
                    zReal, zImage = naive.mandelbrotFunc(
                        zReal, zImage, cReal, cImage)
                    squareOfModulus = zReal * zReal + zImage * zImage
                    # print("i={}, shoule smaller:".format(i), squareOfModulus,
                    #       common.threshold * common.threshold)
                    self.assertTrue(
                        squareOfModulus <= common.threshold * common.threshold)
                    i += 1

                if iotaC != common.iterationLimit:
                    zReal, zImage = naive.mandelbrotFunc(
                        zReal, zImage, cReal, cImage)
                    squareOfModulus = zReal * zReal + zImage * zImage
                    # print("should larger:", squareOfModulus,
                    #       common.threshold * common.threshold)
                    self.assertTrue(
                        squareOfModulus > common.threshold * common.threshold)

    # ensure that all methods have the same `reals`, `images`, and `results`.
    def testAllMethods(self):
        if len(self.calcResultsMethods) < 2:
            return

        print("test methods:", self.calcResultsMethods)

        wrongReals, wrongImages, wrongReals2d, wrongImages2d, wrongResults = wrongCalcResults(
        )

        lastReals, lastImages, lastReals2d, lastImages2d, lastResults = eval(
            self.calcResultsMethods[0])
        for i in (range(1, len(self.calcResultsMethods))):
            print("comparing: {} and {}".format(self.calcResultsMethods[i - 1],
                                                self.calcResultsMethods[i]))
            reals, images, reals2d, images2d, results = eval(
                self.calcResultsMethods[i])
            self.assertTrue(
                np.allclose(lastReals, reals),
                "{} and {} methods, reals are different".format(
                    self.calcResultsMethods[i - 1],
                    self.calcResultsMethods[i]))
            self.assertTrue(
                np.allclose(lastImages, images),
                "{} and {} methods, images are different".format(
                    self.calcResultsMethods[i - 1],
                    self.calcResultsMethods[i]))
            self.assertTrue(
                np.allclose(lastReals2d, reals2d),
                "{} and {} methods, reals2d are different".format(
                    self.calcResultsMethods[i - 1],
                    self.calcResultsMethods[i]))
            self.assertTrue(
                np.allclose(lastImages2d, images2d),
                "{} and {} methods, images2d are different".format(
                    self.calcResultsMethods[i - 1],
                    self.calcResultsMethods[i]))
            self.assertTrue(
                np.allclose(lastResults, results),
                "{} and {} methods, results are different".format(
                    self.calcResultsMethods[i - 1],
                    self.calcResultsMethods[i]))

            # print(lastResults.shape)
            # print(results.shape)
            # for i in range(len(lastResults)):
            #     for j in range(len(lastResults[0])):
            #         if not np.allclose(lastResults[i][j], results[i][j]):
            #             print(i, j, lastResults[i][j], results[i][j])

            self.assertFalse(np.allclose(lastReals, wrongReals))
            self.assertFalse(np.allclose(lastImages, wrongImages))
            self.assertFalse(np.allclose(lastReals2d, wrongReals2d))
            self.assertFalse(np.allclose(lastImages2d, wrongImages2d))
            self.assertFalse(np.allclose(lastResults, wrongResults))

            lastReals, lastImages, lastReals2d, lastImages2d, lastResults = reals, images, reals2d, images2d, results


if __name__ == '__main__':
    unittest.main()
