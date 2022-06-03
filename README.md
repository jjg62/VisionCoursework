# CM30080 - Computer Vision Coursework

Computer Vision team coursework task coded in Python, including the following topics:
- Canny Edge Detection
- Hough Line Transform
- Correlation-Based Template Matching (using normalised cross correlation)
- Multi-Resolution Analysis of Images (to speed up template matching)
- Feature-Based Instance Recognition (using SIFT features)

See the report pdf for more details on implementation decisions, justification of hyper-parameter choices and test results.

## Part 1 - Angle between two lines

The first task involved implementing a program to output the angle between two lines drawn on an image.
The program uses Canny Edge Detection (from the OpenCV library) to detect edge points, and a Hough Transform implemented from scratch to redraw the two lines. This gives their parameters (using the rho/theta parameterisation) which can be easily used to calculate the angle between both lines.

### Usage

Change the path of the 'input' variable on line 7 of Part1/main.py to the desired input image, and run the file. The angle will be printed to the console, and various debug images will be displayed.

## Part 2 - Template matching

The second task involved implementing size invariant template matching using normalised cross correlation. Multi-resolution analysis is used on the input test image to drastically decrease the search space. Gaussian Pyramids are generated for each template image to allow detection of multiple sizes. See the report for analysis of correctness of results.

### Usage

Run Part2/test.py to run a set of automated tests. Alternatively, change the inputName variable at the bottom of Part2/main.py, comment out the relevant lines and run the main file to run the program on a specific test image.

## Part 3 - Feature-based instance recognition

The third task involved using the same dataset as Part 2, but implementing a feature-based appraoch instead. SIFT features are extracted from the dataset, and a bruteforce matcher (with a "ratio test") is used to find the closest feature descriptor for each feature found in the test image.

### Usage

Similarly to Part 2, Part3/test.py can be used to run automated tests, and (the end of) Part3/main.py can be changed to test one specific test image.
