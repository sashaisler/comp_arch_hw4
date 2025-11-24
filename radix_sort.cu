#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define THREADS_PER_BLOCK 256
#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS) // 256
#define RADIX_MASK (RADIX - 1)

// Kernel 1: build histogram for this radix pass
__global__ void buildHistogramKernel(const uint32_t *d_in,
                                     int n,
                                     int shift,
                                     unsigned int *d_hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t val = d_in[idx];
    unsigned int bucket = (val >> shift) & RADIX_MASK;

    // Global histogram with atomics
    atomicAdd(&d_hist[bucket], 1);
}

// Kernel 2: scatter elements into their correct position for this pass
__global__ void scatterKernel(const uint32_t *d_in,
                              uint32_t *d_out,
                              int n,
                              int shift,
                              unsigned int *d_binCounters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t val = d_in[idx];
    unsigned int bucket = (val >> shift) & RADIX_MASK;

    // Each element gets a unique index in its bucket via atomicAdd
    unsigned int pos = atomicAdd(&d_binCounters[bucket], 1);

    d_out[pos] = val;
}

// Host helper: run LSD radix sort (32-bit, 8 bits per pass) on device array
void radixSortGPU(uint32_t *d_data, int n) {
    // Temporary buffers
    uint32_t *d_temp = NULL;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));

    // Histogram and per-pass counters
    unsigned int *d_hist = NULL;
    unsigned int *d_binCounters = NULL;
    CUDA_CHECK(cudaMalloc(&d_hist, RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_binCounters, RADIX * sizeof(unsigned int)));

    // Host-side copies for the prefix sum
    unsigned int h_hist[RADIX];
    unsigned int h_offsets[RADIX];

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // We’ll sort 32-bit integers using 4 passes of 8 bits each
    for (int pass = 0; pass < (32 / RADIX_BITS); ++pass) {
        int shift = pass * RADIX_BITS;

        // 1) Build histogram on the GPU
        CUDA_CHECK(cudaMemset(d_hist, 0, RADIX * sizeof(unsigned int)));
        buildHistogramKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, n, shift,
                                                            d_hist);
        CUDA_CHECK(cudaGetLastError());

        // 2) Copy histogram back to host and build exclusive prefix sum
        CUDA_CHECK(cudaMemcpy(h_hist, d_hist,
                              RADIX * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        // Exclusive scan (prefix sum) on CPU
        unsigned int sum = 0;
        for (int i = 0; i < RADIX; ++i) {
            h_offsets[i] = sum;
            sum += h_hist[i];
        }

        // 3) Copy offsets to device; these will be used as per-bucket counters
        CUDA_CHECK(cudaMemcpy(d_binCounters, h_offsets,
                              RADIX * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // 4) Scatter elements into temp buffer based on this pass’s digit
        scatterKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_data, d_temp, n, shift, d_binCounters);
        CUDA_CHECK(cudaGetLastError());

        // 5) Swap input/output buffers for next pass
        uint32_t *tmp = d_data;
        d_data = d_temp;
        d_temp = tmp;
    }

    // After an even number of passes (4), data is back in d_data.
    // If you want sorted result in original pointer, you’re already good.

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_binCounters));
}

// Simple demo main()
int main(void) {
    const int N = 16;
    uint32_t h_in[N] = {
        329, 457, 657, 839,
        436, 720, 355, 532,
        3,   5,   1,   8,
        10,  2,   7,   6
    };

    printf("Input:\n");
    for (int i = 0; i < N; ++i) {
        printf("%u ", h_in[i]);
    }
    printf("\n");

    uint32_t *d_data = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_data, h_in, N * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Run radix sort on GPU
    radixSortGPU(d_data, N);

    uint32_t h_out[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_data, N * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    printf("Sorted:\n");
    for (int i = 0; i < N; ++i) {
        printf("%u ", h_out[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}

