# Mandelbrot Set Python

This is an assignment of a course about High Performance Computing In Python.

My Python Version is 3.11.2.\
I did not test other versions, but I guess that some other versions may also be OK.

>Note:\
>Some functions (perhaps the parts related to multi-process or multi-threads) should be executed on Linux system.

---
### How to use the Python Unit Test to validate the correctness of this project?
* On a Linux PC or VM, execute `python3 -u unit_tests.py`. If there is no error, the program is correct.
* This test file `unit_tests.py` validates that:
    1. The `naive` method of Mandelbrot Set is correct.
    2. All methods have the same `reals`, `images`, and `results`.
    3. Because of the above 2 points, all methods of Mandelbrot Set are correct.

### How to run this project to get the output data and figure?
1. On a Linux PC or VM, run all cells in the file `gui.ipynb`. Then the `reals`, `images`, and `results` of all methods will be generated in `csv` files. All methods have the same contents in the `csv` files because of the validation in `unit_tests.py`
2. To get the pdf-version figure of Mandelbrot Set, run `python3 -u naive.py` on a PC or VM with GUI (graphical user interface). One figure of Mandelbrot Set is enough in this project, because All methods have the same contents in the `csv` files, which will generate the same figures of Mandelbrot Set.

### How to benchmark this project to get the output data and figure?
1. On a Linux PC or VM, execute `python3 -u benchmark.py` to generate the results of the benchmark in the `csv` file, and also print the results on the screen.
2. On a PC or VM with GUI, execute `python3 -u benchmark_plotting.py` to read the benchmark results from the `csv` file, and plot them in a bar chart.
