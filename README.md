# Canny-Cuda
Canny Edge Detector with CUDA Programming

## Summary of the approach

### CUDA Parallelization Strategy

#### Thread Organization
The main idea is to assign one GPU thread to each pixel in the image. This creates a natural way to divide the work since each pixel can be processed at the same time. The code uses 16×16 thread blocks, which means 256 threads work together in each block. The number of blocks is calculated based on the image size to make sure every pixel gets processed (in this case, a 32 by 32 grid).

#### Memory Management
The implementation uses different types of GPU memory efficiently. Most image data is stored in global memory, which all threads can access. For the min/max finding operation, shared memory is used to speed things up by keeping data closer to the processing units.

### Algorithm Implementation 

#### Gaussian Filtering
This step removes noise from the image by blurring it slightly. Each thread calculates one output pixel by applying a gaussian filter to its area of the input image. The filter size changes automatically based on the sigma parameter - bigger sigma means more blur. Boundary pixels are handled carefully to avoid accessing memory outside the image.

#### Min/Max Finding
To normalize the image later, we need to find the brightest and darkest pixels. This is tricky to do in parallel since we need to compare all pixels. The solution uses a two-step process: first, each block of threads finds its local minimum and maximum using shared memory. Then, the CPU combines all these local results to find the global min and max. This algorithm corresponds to the parallel reduction (sequencial addressing).

#### Normalization
Once we know the min and max values, each thread can independently scale its pixel to the standard 0-255 range. This is perfectly parallel since no communication between threads is needed.

#### Gradient Computation
This step finds edges by looking for rapid changes in brightness. It uses two Sobel filters - one for horizontal changes and one for vertical changes. The same convolution function is reused for both directions by just changing the filter coefficients. After both gradients are calculated, another kernel combines them to get the final gradient strength.

#### Non-Maximum Suppression
This step makes thick edges thinner by keeping only the strongest pixels along each edge direction. Each thread looks at its pixel's gradient direction and compares its strength with neighboring pixels in that direction. Only pixels that are stronger than their neighbors survive this step.

#### Hysteresis Thresholding
The final step uses two thresholds to decide which pixels are really edges. Strong pixels (above the high threshold) are automatically kept as edges. Weak pixels (above the low threshold) are only kept if they connect to strong edges. The first part is easy to parallelize, but the second part requires multiple passes through the image until no more changes happen.

### Key Optimizations

#### Computational Efficiency
The min/max operation uses a smart reduction technique that works like a tournament - pairs of values are compared and winners advance to the next round. Shared memory keeps intermediate results on the GPU chip, avoiding slower global memory access.

## Setup instructions

Make sure you're in the project root:
```bash
cd cuda-canny
```

Compile using make:
```bash
make
```

## Usage details

Run the executable from the ~/cuda-canny directory:

```bash
./canny [-h] [-d device] [-i inputfile] [-o outputfile] [-r referenceFile] [-s sigma] [-n tmin] [-x tmax]

```

### Parameters:
| Option | Description |
|--------|-------------|
| `-h`   | Display help message |
| `-d`   | CUDA device ID (default: 0) |
| `-i`   | Input PGM image file (default: `images/lake.pgm`) |
| `-o`   | Output image file from device (default: `out.pgm`) |
| `-r`   | Reference output file from host (default: `reference.pgm`) |
| `-s`   | Sigma value for Gaussian blur (default: `1.0`) |
| `-n`   | Minimum threshold (`tmin`, default: `45`) |
| `-x`   | Maximum threshold (`tmax`, default: `50`) |

### Example:
```bash
./canny -i images/lake.pgm -o result.pgm -s 1.4 -n 30 -x 60
```

## Directory Structure

```
~/cuda-canny/
├── images/
│   └── lake.pgm        # Example input image
├── out.pgm             # Output after edge detection
├── reference.pgm       # Optional reference output
├── canny               # Compiled executable
├── Makefile
└── src/                # Source files
```

## Performance Analysis 

To evaluate the performance of the CUDA-accelerated Canny Edge Detector, we ran the implementation 100 times with default parameters and compared execution times between the CPU (host) and GPU (device) versions.

### Execution Time

| Metric                   | Value         |
|--------------------------|---------------|
| Avg. Host Time           | 32.92 ms      |
| Avg. Device Time         | 1.34 ms       |

### Speedup

```
Speedup = CPU_Time / GPU_Time
        = 32.92 / 1.34 ≈ 24.6×
```

### Output Quality

**Pixel difference**: 1 / 262144 (≈ 0.0004%)

The device and host output images are nearly identical, demonstrating correctness of the CUDA implementation.

### Sigma Variation Analysis

To evaluate how the Gaussian blur parameter (`sigma`) affects performance and accuracy, we ran the program 10 times for each of several sigma values while keeping other parameters default. We recorded the average host time, device time, speedup, and pixel difference between host and device outputs.

| Sigma | Avg Host Time (ms) | Avg Device Time (ms) | Speedup | Avg Diff Pixels |
|-------|--------------------|-----------------------|---------|-----------------|
| 0.5   | 40.20              | 1.27                  | 31.70×  | 3               |
| 1.0   | 32.12              | 1.34                  | 23.92×  | 1               |
| 1.5   | 36.61              | 1.34                  | 27.32×  | 1               |
| 2.0   | 44.68              | 1.41                  | 31.62×  | 6               |
| 2.5   | 51.06              | 1.55                  | 33.02×  | 3               |
| 3.0   | 61.52              | 1.70                  | 36.27×  | 0               |
