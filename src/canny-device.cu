// CLE 24'25

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define MAX_BRIGHTNESS 255
#define THREADS 16

// Use int instead `unsigned char' so that we can
// store negative values.
typedef int pixel_t;

// convolution of in image to out image using kernel of kn width
__global__ void convolution_device(
        const pixel_t* in, 
        pixel_t* out, 
        const float *kernel, 
        const int Nx, 
        const int Ny, 
        const int kernel_size
        ) {
    //center of kernel in both dimensions
    int center = (kernel_size - 1)/2;

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float pixel = 0.0;

    if (ix >= center && iy >= center && ix < Nx - center && iy < Ny - center){
        for (int ki = 0; ki<kernel_size; ki++)
            for (int kj = 0; kj<kernel_size; kj++){
                pixel+=in[(ki + iy - center) * Nx + kj + ix - center] * kernel[ki*kernel_size + kj];
            }
    }
    out[iy * Nx + ix] = (pixel_t) pixel;
}

// determines min and max of in image
__global__ void min_max_device(
        const pixel_t *in, 
        const int Nx, 
        const int Ny, 
        pixel_t *pmin, 
        pixel_t *pmax
        ) {
    extern __shared__ pixel_t sminmax[];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * Nx + (blockIdx.y * blockDim.y + threadIdx.y);

    pixel_t *sdata_min = sminmax;
    pixel_t *sdata_max = sminmax + blockDim.x * blockDim.y;

    // each thread loads one element from global to shared memory
    sdata_min[tid] = in[idx];
    sdata_max[tid] = in[idx];

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x * blockDim.y/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_min[tid] = min(sdata_min[tid], sdata_min[tid + s]);
            sdata_max[tid] = max(sdata_max[tid], sdata_max[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) { 
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        pmin[bid] = sdata_min[0];
        pmax[bid] = sdata_max[0];
    }
}

// normalizes inout image using min and max values
__global__ void normalize_device(
        pixel_t *inout,
        const int Nx, 
        const int Ny, 
        const int kernel_size,
        const int min,
        const int max
        ) {
    //center of kernel in both dimensions
    int center = (kernel_size - 1)/2;

    // pixel coordinates
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= center && iy >= center && ix < Nx - center && iy < Ny - center){
        int idx = ix + Nx * iy;
        pixel_t pixel = MAX_BRIGHTNESS * ((int)inout[idx] -(float)min) / ((float)max - (float)min);
        inout[idx] = pixel;
    }
}

// Canny non-maximum suppression
__global__ void non_maximum_supression_device(
        const pixel_t* __restrict__ after_Gx, 
        const pixel_t* __restrict__ after_Gy, 
        const pixel_t* __restrict__ G, 
        pixel_t* nms, 
        const int Nx, 
        const int Ny
        ) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= 1 && ix < Nx - 1 && iy >= 1 && iy < Ny - 1) {
        const int idx = ix + Nx * iy;
        const int nn = idx - Nx;
        const int ss = idx + Nx;
        const int ww = idx + 1;
        const int ee = idx - 1;
        const int nw = nn + 1;
        const int ne = nn - 1;
        const int sw = ss + 1;
        const int se = ss - 1;

        const float dir = (float)(fmod(atan2f(after_Gy[idx],
                        after_Gx[idx]) + M_PI,
                    M_PI) / M_PI) * 8;

        if (((dir <= 1 || dir > 7) && G[idx] > G[ee] && G[idx] > G[ww]) ||  // 0 deg
                ((dir > 1 && dir <= 3) && G[idx] > G[nw] && G[idx] > G[se]) ||  // 45 deg
                ((dir > 3 && dir <= 5) && G[idx] > G[nn] && G[idx] > G[ss]) ||  // 90 deg
                ((dir > 5 && dir <= 7) && G[idx] > G[ne] && G[idx] > G[sw])     // 135 deg
           )   
            nms[idx] = G[idx];
        else
            nms[idx] = 0;
    }
}

// edges found in first pass for nms > tmax
__global__ void first_edges_device(
        const pixel_t *nms, 
        pixel_t *reference, 
        const int Nx, 
        const int Ny, 
        const int tmax
        ) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= 1 && ix < Nx - 1 && iy >= 1 && iy < Ny - 1) {
        int idx = ix + Nx * iy;
        if (nms[idx] >= tmax) {
            reference[idx] = MAX_BRIGHTNESS;
        }
    }
}

// edges found in after first passes for nms > tmin && neighbor is edge
__global__ void hysteresis_edges_device(
        const pixel_t *nms, 
        pixel_t *reference,
        const int Nx, 
        const int Ny, 
        const int tmin, 
        int *dchanged
        ) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= 1 && ix < Nx - 1 && iy >= 1 && iy < Ny - 1) {
        int idx = ix + Nx * iy;
        if (nms[idx] >= tmin && reference[idx] == 0) {
            int neighbors[8] = {
                idx - Nx,         // N
                idx + Nx,         // S
                idx + 1,          // E
                idx - 1,          // W
                idx - Nx + 1,     // NE
                idx - Nx - 1,     // NW
                idx + Nx + 1,     // SE
                idx + Nx - 1      // SW
            };
            for (int k = 0; k < 8; k++) {
                if (reference[neighbors[k]] != 0) {
                    // Only the thread that successfully flips 0 to MAX_BRIGHTNESS
                    // sets the changed flag, because the others will see MAX_BRIGHTNESS
                    int old = atomicCAS(&reference[idx], 0, MAX_BRIGHTNESS);
                    if (old == 0) { *dchanged = 1; };
                    break;
                }
            }
        }
    }
}

__global__ void merge_gradients(
        const pixel_t* __restrict__ after_Gx,
        const pixel_t* __restrict__ after_Gy,
        pixel_t* __restrict__ G,
        int Nx,
        int Ny
        ) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= 1 && ix < Nx - 1 && iy >= 1 && iy < Ny - 1) {
        int idx = iy * Nx + ix;

        float gx = static_cast<float>(after_Gx[idx]);
        float gy = static_cast<float>(after_Gy[idx]);

        G[idx] = static_cast<pixel_t>(hypotf(gx, gy));
    }
}

// canny edge detector code to run on the GPU
__host__ void cannyDevice(
        const int *h_idata, 
        const int Nx, 
        const int Ny,
        const int tmin, 
        const int tmax,
        const float sigma,
        int *reference
        ) {
    int BLOCKS = (Nx + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    const float Gx[] = {-1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};

    const float Gy[] = { 1, 2, 1,
        0, 0, 0,
        -1,-2,-1};

    float *d_G;
    pixel_t *d_in, *after_Gx, *after_Gy, *nms, *d_reference;

    cudaMalloc(&d_in, Nx * Ny * sizeof(pixel_t));
    cudaMalloc(&d_reference, Nx * Ny * sizeof(pixel_t));
    cudaMalloc(&after_Gx, Nx * Ny * sizeof(pixel_t));
    cudaMalloc(&after_Gy, Nx * Ny * sizeof(pixel_t));
    cudaMalloc(&nms, Nx * Ny * sizeof(pixel_t));
    cudaMalloc(&d_G, 9 * sizeof(float));

    if (after_Gx == NULL || after_Gy == NULL ||
            nms == NULL || d_reference == NULL ) {
        fprintf(stderr, "canny_edge_detection:"
                " Failed memory allocation(s).\n");
        exit(1);
    }

    /*
     * Gaussian Filter Process:
     * http://www.songho.ca/dsp/cannyedge/cannyedge.html
     * determine size of kernel (odd #)
     * 0.0 <= sigma < 0.5 : 3
     * 0.5 <= sigma < 1.0 : 5
     * 1.0 <= sigma < 1.5 : 7
     * 1.5 <= sigma < 2.0 : 9
     * 2.0 <= sigma < 2.5 : 11
     * 2.5 <= sigma < 3.0 : 13 ...
     * kernelSize = 2 * int(2*sigma) + 3;
     */

    const int n = 2 * (int)(2 * sigma) + 3;
    const float mean = (float)floor(n / 2.0);
    float kernel[n * n];

    float *d_kernel;
    cudaMalloc(&d_kernel, n * n * sizeof(float));

    fprintf(stderr, "gaussian_filter: kernel size %d, sigma=%g\n",
            n, sigma);
    size_t c = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            kernel[c] = expf(-0.5 * (powf((i - mean) / sigma, 2.0) +
                        powf((j - mean) / sigma, 2.0)))
                / (2 * M_PI * sigma * sigma);
            c++;
        }

    // copy kernel and initial image to device
    cudaMemcpy(d_kernel, &kernel, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, h_idata, Nx*Ny*sizeof(pixel_t), cudaMemcpyHostToDevice);

    // for some reason when doing convolution in gpu gives 1 or 2 different pixels in some images
    convolution_device<<<grid_dim, block_dim>>>(d_in, d_reference, d_kernel, Nx, Ny, n);

    pixel_t h_min[BLOCKS * BLOCKS], h_max[BLOCKS * BLOCKS];

    pixel_t *d_minmax;
    cudaMalloc(&d_minmax, 2 * BLOCKS * BLOCKS * sizeof(pixel_t));

    pixel_t* d_min = d_minmax;
    pixel_t* d_max = d_minmax + BLOCKS * BLOCKS;

    // use parallel reduction to find min and max
    min_max_device<<<grid_dim, block_dim, 2 * THREADS * THREADS * sizeof(pixel_t)>>>(d_reference, Nx, Ny, d_min, d_max);

    cudaMemcpy(h_min, d_min, BLOCKS * BLOCKS * sizeof(pixel_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max, d_max, BLOCKS * BLOCKS * sizeof(pixel_t), cudaMemcpyDeviceToHost);

    pixel_t pmax = h_max[0], pmin = h_min[0];
    for (int i = 1; i < BLOCKS * BLOCKS; i++) {
        pmin = min(pmin, h_min[i]);
        pmax = max(pmax, h_max[i]);
    }

    normalize_device<<<grid_dim, block_dim>>>(d_reference, Nx, Ny, n, pmin, pmax);

    /* -- -- -- -- -- -- -- -- */

    // Gradient along x
    cudaMemcpy(d_G, Gx, 9 * sizeof(float), cudaMemcpyHostToDevice);
    convolution_device<<<grid_dim, block_dim>>>(d_reference, after_Gx, d_G, Nx, Ny, 3);

    // Gradient along y
    cudaMemcpy(d_G, Gy, 9 * sizeof(float), cudaMemcpyHostToDevice);
    convolution_device<<<grid_dim, block_dim>>>(d_reference, after_Gy, d_G, Nx, Ny, 3);

    // Merging gradients
    merge_gradients<<<grid_dim, block_dim>>>(after_Gx, after_Gy, d_reference, Nx, Ny);

    // Non-maximum suppression, straightforward implementation.
    non_maximum_supression_device<<<grid_dim, block_dim>>>(after_Gx, after_Gy, d_reference, nms, Nx, Ny);

    cudaMemset(d_reference, 0, Nx * Ny * sizeof(pixel_t));
    first_edges_device<<<grid_dim, block_dim>>>(nms, d_reference, Nx, Ny, tmax);

    // edges with nms >= tmin && neighbor is edge
    int changed;
    int *d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    do {
        changed = false;
        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);
        hysteresis_edges_device<<<grid_dim, block_dim>>>(nms, d_reference, Nx, Ny, tmin, d_changed);
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (changed==true);

    cudaMemcpy(reference, d_reference, Nx * Ny * sizeof(pixel_t), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_kernel);
    cudaFree(d_minmax);
    cudaFree(after_Gx);
    cudaFree(after_Gy);
    cudaFree(d_G);
    cudaFree(nms);
    cudaFree(d_reference);
    cudaFree(d_changed);
}
