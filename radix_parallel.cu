#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
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

#define BLOCK_SIZE 256          // also number of threads per block
#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS) // 256
#define RADIX_MASK (RADIX - 1)

// ---------------- Kernel A: per-block histogram ----------------
//
// Each block handles up to BLOCK_SIZE elements (contiguous).
// Builds a shared-memory histogram of 256 buckets, then writes it
// to global memory at d_blockHist[blockIdx * RADIX + bucket].

__global__
void buildBlockHistKernel(const uint32_t *d_in,
                          int n,
                          int shift,
                          unsigned int *d_blockHist)
{
    __shared__ unsigned int s_hist[RADIX];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * BLOCK_SIZE;
    int idx = blockStart + tid;

    // Initialize shared histogram
    for (int b = tid; b < RADIX; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    // Each thread processes at most one element in this simple version
    if (idx < n) {
        uint32_t v = d_in[idx];
        unsigned int bucket = (v >> shift) & RADIX_MASK;
        atomicAdd(&s_hist[bucket], 1);
    }
    __syncthreads();

    // Write shared histogram to global memory
    if (tid < RADIX) {
        d_blockHist[blockIdx.x * RADIX + tid] = s_hist[tid];
    }
}

// ---------------- Kernel B: stable scatter per block ----------------
//
// Each block scatters its chunk of elements into the correct positions.
// We use per-block offsets (precomputed on host) and local counters
// in shared memory. Only thread 0 does the actual scattering loop
// (simple and stable); all threads help init localCount.

__global__
void scatterKernel(const uint32_t *d_in,
                   uint32_t *d_out,
                   int n,
                   int shift,
                   const unsigned int *d_blockOffsets)
{
    __shared__ unsigned int localCount[RADIX];

    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int blockStart = blockId * BLOCK_SIZE;
    int blockEnd = blockStart + BLOCK_SIZE;
    if (blockEnd > n) blockEnd = n;

    // Initialize local counts
    for (int b = tid; b < RADIX; b += blockDim.x) {
        localCount[b] = 0;
    }
    __syncthreads();

    // Pointer to this block's per-bucket base offsets
    const unsigned int *blockOffsetBase = d_blockOffsets + blockId * RADIX;

    // To keep it simple and stable, only thread 0 does the scatter loop
    if (tid == 0) {
        for (int idx = blockStart; idx < blockEnd; ++idx) {
            uint32_t v = d_in[idx];
            unsigned int bucket = (v >> shift) & RADIX_MASK;

            unsigned int pos = blockOffsetBase[bucket] + localCount[bucket];
            localCount[bucket]++;

            d_out[pos] = v;
        }
    }
}

// ---------------- 8-bit-per-pass LSD radix sort ----------------

void radixSortGPU8bit(uint32_t *d_data, int n, float *elapsed_ms_out)
{
    // Temporary array (ping-pong)
    uint32_t *d_temp = NULL;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));

    // Number of blocks (each block handles BLOCK_SIZE elements)
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Per-block histograms and offsets (size: numBlocks * RADIX)
    unsigned int *d_blockHist = NULL;
    unsigned int *d_blockOffsets = NULL;
    CUDA_CHECK(cudaMalloc(&d_blockHist,    numBlocks * RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, numBlocks * RADIX * sizeof(unsigned int)));

    // Host buffers for hist and offsets
    unsigned int *h_blockHist    = (unsigned int*)malloc(numBlocks * RADIX * sizeof(unsigned int));
    unsigned int *h_blockOffsets = (unsigned int*)malloc(numBlocks * RADIX * sizeof(unsigned int));
    unsigned int  h_bucketTotals[RADIX];
    unsigned int  h_bucketBase[RADIX];

    if (!h_blockHist || !h_blockOffsets) {
        fprintf(stderr, "Host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // 4 passes for 32-bit integers, 8 bits at a time
    for (int pass = 0; pass < 32 / RADIX_BITS; ++pass) {
        int shift = pass * RADIX_BITS;

        // 1) Build per-block histograms on GPU
        buildBlockHistKernel<<<numBlocks, BLOCK_SIZE>>>(d_data, n, shift, d_blockHist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2) Copy per-block histograms to host
        CUDA_CHECK(cudaMemcpy(h_blockHist, d_blockHist,
                              numBlocks * RADIX * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        // 3) Compute total counts per bucket
        for (int b = 0; b < RADIX; ++b) {
            h_bucketTotals[b] = 0;
        }
        for (int blk = 0; blk < numBlocks; ++blk) {
            for (int b = 0; b < RADIX; ++b) {
                h_bucketTotals[b] += h_blockHist[blk * RADIX + b];
            }
        }

        // 4) Compute global bucket bases (exclusive prefix sum on buckets)
        unsigned int sum = 0;
        for (int b = 0; b < RADIX; ++b) {
            h_bucketBase[b] = sum;
            sum += h_bucketTotals[b];
        }

        // 5) Compute per-block offsets for each bucket
        //    For each bucket, we walk blocks in order and assign ranges
        for (int b = 0; b < RADIX; ++b) {
            unsigned int running = h_bucketBase[b];
            for (int blk = 0; blk < numBlocks; ++blk) {
                int idx = blk * RADIX + b;
                h_blockOffsets[idx] = running;
                running += h_blockHist[idx];
            }
        }

        // 6) Copy per-block offsets back to device
        CUDA_CHECK(cudaMemcpy(d_blockOffsets, h_blockOffsets,
                              numBlocks * RADIX * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // 7) Scatter into temp buffer
        scatterKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_data, d_temp, n, shift, d_blockOffsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 8) Swap d_data and d_temp for next pass
        uint32_t *tmp = d_data;
        d_data = d_temp;
        d_temp = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms_out) *elapsed_ms_out = elapsed_ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free temporary GPU memory
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaFree(d_blockOffsets));

    // NOTE: final sorted data now lives in d_data (the pointer that was passed in).
    // In our main() we keep track of that pointer, so this is fine.

    free(h_blockHist);
    free(h_blockOffsets);
}

// -------------------------- main --------------------------

int main(void)
{
    const int N = 1 << 25; // 1M elements, tweak as needed

    uint32_t *h_in = (uint32_t*)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t*)malloc(N * sizeof(uint32_t));
    if (!h_in || !h_out) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Fill with random data
    for (int i = 0; i < N; ++i) {
        h_in[i] = (uint32_t)rand();
    }

    uint32_t *d_data = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_data, h_in, N * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    float elapsed_ms = 0.0f;
    radixSortGPU8bit(d_data, N, &elapsed_ms);

    printf("8-bit-per-pass GPU radix sort time: %.3f ms\n", elapsed_ms);

    // Copy back result to host and verify
    CUDA_CHECK(cudaMemcpy(h_out, d_data, N * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    int fail_idx = -1;
    for (int i = 1; i < N; ++i) {
        if (h_out[i - 1] > h_out[i]) {
            fail_idx = i;
            break;
        }
    }

    if (fail_idx == -1) {
        printf("Sort verification PASSED\n");
    } else {
        printf("Sort verification FAILED at index %d (values %u, %u)\n",
               fail_idx, h_out[fail_idx - 1], h_out[fail_idx]);
    }

    CUDA_CHECK(cudaFree(d_data));
    free(h_in);
    free(h_out);

    return 0;
}

